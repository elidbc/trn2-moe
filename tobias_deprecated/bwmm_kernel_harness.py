import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import torch_neuronx
import torch_xla.core.xla_model as xm

from kernel_bwmm import kernel
from mistral_moe import MixtralConfig, MixtralRouter

# define constants
INPUT_MODEL_PATH = "weights/uncompiled_model_weights/mistral_moe_new.pt"
OUTPUT_MODEL_PATH = "weights/compiled_model_weights/mistral_moe_kernelized.pt"
INPUT_TENSOR_PATH = "weights/input_weights/prefill_b1_s4096.pt"
OUTPUT_TENSOR_PATH = "weights/output_weights/out_prefill_b1_s4096.pt"
TOKENS_PER_BLOCK = 128
MAX_BLOCKS_PER_EXPERT = 12 # TODO: think about this more later

class KernelizedMoE(nn.Module):
    def __init__(self, config, tokens_per_block=TOKENS_PER_BLOCK):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts       
        self.top_k = config.topk         
        self.router = MixtralRouter(config)
        self.tokens_per_block = tokens_per_block

        # The Experts (Fixed: proper nn.Parameter syntax and transposed shapes)
        # note that these are transposed relative to the expert matricies in mistral_moe.py
        self.w1 = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, config.intermediate_size))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, config.intermediate_size, config.hidden_size))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, config.intermediate_size))
        self.act_fn = nn.SiLU()         # technically dont need this

        

    def generate_static_blocks(self, selected_experts: torch.Tensor, max_blocks: int): 
        selected_experts = selected_experts.to(torch.long)
        num_tokens, top_k = selected_experts.shape
        device = selected_experts.device
        block_size = self.tokens_per_block
        
        # 1. Count tokens per expert to calculate block offsets statically
        # One-hot encode routing: [num_tokens, top_k, num_experts]
        one_hot = F.one_hot(selected_experts, num_classes=self.num_experts)
        
        # Sum over tokens and top_k to get the exact count per expert: [num_experts]
        expert_counts = one_hot.sum(dim=(0, 1)) 
        
        # Calculate how many blocks each expert needs, and their starting block index globally
        expert_blocks = (expert_counts + block_size - 1) // block_size 
        expert_block_offsets = torch.cumsum(expert_blocks, dim=0) - expert_blocks 
        
        # 2. Get the "queue position" for each specifically routed token
        # Sum over top_k to get an integer mask: [num_tokens, num_experts]
        expert_mask = one_hot.sum(dim=1)
        token_positions = torch.cumsum(expert_mask, dim=0) - 1 
        
        # Gather ONLY the queue positions for the experts we actually selected
        # Shape: [num_tokens, top_k]
        selected_positions = torch.gather(token_positions, 1, selected_experts) 
        
        # 3. Calculate target global block coordinates for each token
        # Gather the global block offsets for our selected experts: [num_tokens, top_k]
        expanded_offsets = expert_block_offsets.unsqueeze(0).expand(num_tokens, -1)
        selected_block_offsets = torch.gather(expanded_offsets, 1, selected_experts) 
        
        # Calculate exactly which global block and offset the token belongs to
        global_block_indices = selected_block_offsets + (selected_positions // block_size) 
        offsets_in_block = selected_positions % block_size 
        
        # 4. Pre-allocate the globally flat static output tensors [cite: 89, 90]
        # flattened_blocks: [max_blocks, block_size] filled with -1 padding [cite: 85]
        # expert_ids: [max_blocks] filled with -1
        flat_flattened_blocks = torch.full((max_blocks * block_size,), -1, dtype=torch.long, device=device)
        expert_ids = torch.full((max_blocks,), -1, dtype=torch.long, device=device)
        
        # 5. Scatter the data using flattened 1D indices
        # We flatten all our coordinate tensors to shape [num_tokens * top_k]
        flat_global_blocks = global_block_indices.view(-1) 
        flat_offsets = offsets_in_block.view(-1) 
        flat_expert_ids = selected_experts.view(-1) 
        
        # Token IDs grid: [0, 0, 1, 1, 2, 2...] for top_k=2
        flat_token_ids = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        
        # Map the 2D block coordinates to a flat 1D index
        flat_target_indices = flat_global_blocks * block_size + flat_offsets
        
        # Scatter the tokens into the blocks
        flat_flattened_blocks[flat_target_indices] = flat_token_ids
        flattened_blocks = flat_flattened_blocks.view(max_blocks, block_size)
        
        # Scatter the expert IDs (multiple tokens write the same expert ID to the same block index, which is safe)
        expert_ids[flat_global_blocks] = flat_expert_ids

        # new return statement
        expert_block_offsets = expert_block_offsets.to(torch.int32)             # where the blocks for each expert begin and end in the flattend blocks matrix
        expert_blocks = expert_blocks.to(torch.int32)                           # number of blocks per expert
        flattened_blocks = flattened_blocks.to(torch.int32)                     # max blocks x block size --- > stores the blocks as row vectors
        expert_blocks = torch.clamp(expert_blocks, max=MAX_BLOCKS_PER_EXPERT)   # clamp
        return flattened_blocks, expert_block_offsets, expert_blocks
        
        # return flattened_blocks, expert_ids

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim) 
        routing_weights, selected_experts = self.router(hidden_states)
        output = torch.zeros_like(hidden_states)

        # Compute max block bound (this will be static at compile time)
        num_tokens = hidden_states.shape[0]
        total_routed_tokens = num_tokens * self.top_k
        active_experts = min(self.num_experts, total_routed_tokens)
        bound_1 = math.floor((total_routed_tokens + (active_experts * (self.tokens_per_block - 1))) / self.tokens_per_block)
        bound_2 = self.num_experts * math.ceil(num_tokens / self.tokens_per_block)
        max_blocks = min(bound_1, bound_2)

        # compute block assignments
        flattened_blocks, expert_block_offsets, expert_block_counts = self.generate_static_blocks(selected_experts, max_blocks)

        # transpose input tensor so it works with nc_matmul
        hidden_states = hidden_states.t().contiguous()

        # run kernel
        kernel(
            hidden_states, 
            self.w1, 
            self.w3,
            self.w2, 
            routing_weights, 
            flattened_blocks, 
            expert_block_offsets,
            expert_block_counts,
            output,
            MAX_BLOCKS_PER_EXPERT
        )

        output = output.view(batch_size, sequence_length, hidden_dim)
        selected_experts = selected_experts.reshape(batch_size, sequence_length, self.top_k)
        return output, selected_experts


def compile_model():
    print("Initializing Kernelized MoE block...")
    moe_block = KernelizedMoE(MixtralConfig())
    
    if os.path.exists(INPUT_MODEL_PATH):
        print(f"Loading model weights from '{INPUT_MODEL_PATH}'...")
        state_dict = torch.load(INPUT_MODEL_PATH, map_location="cpu", weights_only=True)
        # Intercept and transpose before loading
        for key in ['w1', 'w2', 'w3']:
            if key in state_dict:
                state_dict[key] = state_dict[key].transpose(1, 2).contiguous()
        moe_block.load_state_dict(state_dict)
    else:
        print(f"Error: Model weights not found at {INPUT_MODEL_PATH}.")
        return
        
    # Cast to bfloat16 and set to eval mode
    moe_block = moe_block.to(dtype=torch.bfloat16)
    moe_block.eval()
    
    with torch.no_grad(): 
        # Load input and cast to bfloat16
        input_tensor = torch.load(INPUT_TENSOR_PATH, map_location="cpu", weights_only=True)
        input_tensor = input_tensor.to(dtype=torch.bfloat16)
        
        # 1. Trace the model if we haven't seen this shape yet
        print(f"--- Tracing Neuron graph for shape {input_tensor.shape} ---")
            
        # torch_neuronx.trace compiles the model specifically for this tensor shape
        compiled_model = torch_neuronx.trace(moe_block, input_tensor)
        print("--- Tracing complete! ---\n")
        
        # save compiled model 
        print(f"Saving compiled model locally to {OUTPUT_MODEL_PATH}...")
        torch.jit.save(compiled_model, OUTPUT_MODEL_PATH)

    return compiled_model
        

def test_kernel():
    # load in the input tensor
    input_tensor = torch.load(INPUT_TENSOR_PATH)
    input_tensor = input_tensor.to(dtype=torch.bfloat16, device="cpu")

    # create inputs for the kernel and send them to xla
    #device = xm.xla_device()

    # Run NKI kernel using pytorch framework
    model = compile_model()
    kernel_output_tensor, selected_experts = model(input_tensor)
    kernel_output_tensor = kernel_output_tensor.cpu()

    # Load pre-computed output tensor (i.e. the output of the precompiled MoE model on this specific input)
    reference_output_tensor = torch.load(OUTPUT_TENSOR_PATH)
    reference_output_tensor = reference_output_tensor.to(dtype=torch.bfloat16, device="cpu")

    # Compare results
    print("Checking correctness of the compiled kernel...")
    
    # Calculate absolute and relative differences
    # Cast to float32 for metric calculation to avoid bfloat16 overflow/underflow artifacts
    out_f32 = kernel_output_tensor.to(torch.float32)
    ref_f32 = reference_output_tensor.to(torch.float32)
    
    abs_diff = torch.abs(out_f32 - ref_f32)
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()
    
    # Add a tiny epsilon to the denominator to prevent division by zero
    epsilon = 1e-7
    rel_diff = abs_diff / (torch.abs(ref_f32) + epsilon)
    max_rel_diff = torch.max(rel_diff).item()
    mean_rel_diff = torch.mean(rel_diff).item()
    
    # Define our standard tolerances
    atol = 1e-4
    rtol = 1e-2
    
    # Calculate exactly how many elements exceed the allclose formula:
    # absolute(a - b) <= (atol + rtol * absolute(b))
    tolerance_threshold = atol + rtol * torch.abs(ref_f32)
    mismatched_mask = abs_diff > tolerance_threshold
    mismatched_elements = torch.sum(mismatched_mask).item()
    total_elements = ref_f32.numel()
    mismatch_percentage = (mismatched_elements / total_elements) * 100
    
    print("-" * 40)
    print(f"Max Absolute Difference:  {max_abs_diff:.6f}")
    print(f"Mean Absolute Difference: {mean_abs_diff:.6f}")
    print(f"Max Relative Difference:  {max_rel_diff:.6f}")
    print(f"Mean Relative Difference: {mean_rel_diff:.6f}")
    print(f"Mismatched Elements:      {mismatched_elements} / {total_elements} ({mismatch_percentage:.4f}%)")
    print("-" * 40)

    if mismatched_elements == 0:
        print("Result: NKI and Torch match! ✅")
    else:
        print("Result: NKI and Torch differ. ❌")


if __name__ == "__main__":
    test_kernel()