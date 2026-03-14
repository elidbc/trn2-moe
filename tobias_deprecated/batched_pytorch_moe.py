import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class BatchedMixtralExperts(nn.Module):
    """Collection of expert weights stored as contiguous 3D tensors for hardware efficiency."""
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size

        # Stacked 3D tensors for all experts. 
        # gate_up_proj combines W1 (gate) and W3 (up) for the SwiGLU activation.
        # Shape: (num_experts, hidden_dim, 2 * intermediate_dim)
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, 2 * self.intermediate_dim)
        )
        
        # down_proj represents W2.
        # Shape: (num_experts, intermediate_dim, hidden_dim)
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim)
        )
        
        self.act_fn = nn.SiLU()

        # Initialize weights
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Initialize the output tensor with zeros
        final_hidden_states = torch.zeros_like(hidden_states)

        # 1. Create a one-hot mask to map tokens to their assigned experts
        # Shape: (num_experts, batch * seq_len, top_k)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 0, 1)

        # 2. Process each expert (mimicking Grouped GEMM dispatch)
        # In a custom hardware kernel, this loop is replaced by parallel on-device dispatch
        for expert_idx in range(self.num_experts):
            # Find the indices of tokens routed to the current expert
            token_idx, top_k_pos = torch.where(expert_mask[expert_idx])

            # Skip execution if no tokens were routed to this expert
            if token_idx.shape[0] == 0:
                continue

            # Extract the actual token embeddings assigned to this expert
            current_state = hidden_states[token_idx]

            # --- Batched Matrix Multiplications ---
            
            # 1. Gate and Up Projection
            # current_state: (num_tokens, hidden_dim)
            # weight: (hidden_dim, 2 * intermediate_dim)
            gate_up_result = torch.matmul(current_state, self.gate_up_proj[expert_idx])

            # Split into gate (W1) and up (W3) components
            gate, up = gate_up_result.chunk(2, dim=-1)

            # SwiGLU Activation
            current_hidden_states = self.act_fn(gate) * up

            # 2. Down Projection (W2)
            # current_hidden_states: (num_tokens, intermediate_dim)
            # weight: (intermediate_dim, hidden_dim)
            current_hidden_states = torch.matmul(current_hidden_states, self.down_proj[expert_idx])

            # Weight the expert's output by the routing probability
            routing_weights_for_expert = top_k_weights[token_idx, top_k_pos].unsqueeze(-1)
            current_hidden_states = current_hidden_states * routing_weights_for_expert

            # Accumulate the results back into the final output tensor
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class BatchedMixtralSparseMoeBlock(nn.Module):
    """The MoE Routing Block using batched expert weights."""
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # The Router / Gating Network
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        
        # The Batched Experts
        self.experts = BatchedMixtralExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Flatten batch and sequence dimensions for easier routing
        hidden_states = hidden_states.view(-1, hidden_dim) 
        
        # 1. Get router logits and probabilities
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        
        # 2. Select top-k (top-2) experts per token
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # 3. Normalize routing weights so the selected top-k sum to 1
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 4. Dispatch to the batched experts
        final_hidden_states = self.experts(hidden_states, selected_experts, routing_weights)

        # Reshape back to the original dimensions
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states, router_logits




# --- New Additional Code ---

class MixtralConfig:
    """Mock config class to hold the dimensions required by the MoE block."""
    def __init__(self):
        # Using standard Mixtral 8x7B hidden dimensions. 
        # Note: If you OOM on a smaller GPU, cut these down (e.g., 1024 and 4096).
        self.hidden_size = 4096 
        self.intermediate_size = 14336
        self.num_local_experts = 8
        self.num_experts_per_tok = 2

        
def main():
    config = MixtralConfig()
    # Use a new filename so we don't conflict with the old ModuleList weights
    weights_path = "batched_mixtral_moe_dummy_weights.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # 1. Instantiate the Batched MoE block
    moe_block = BatchedMixtralSparseMoeBlock(config)
    
    # 2. Save/Load Weights Logic (Updated for batched tensors)
    if os.path.exists(weights_path):
        print(f"Loading existing weights from '{weights_path}'...")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        moe_block.load_state_dict(state_dict)
    else:
        print(f"Generating new dummy weights and saving to '{weights_path}'...")
        torch.save(moe_block.state_dict(), weights_path)
        
    # 3. Cast to bfloat16 and move to device
    moe_block = moe_block.to(device=device, dtype=torch.bfloat16)
    moe_block.eval() 
    
    # 4. Create Dummy Input
    batch_size = 2
    seq_len = 16
    print(f"Creating dummy input of shape ({batch_size}, {seq_len}, {config.hidden_size}) in bfloat16...")
    
    dummy_input = torch.randn(
        batch_size, seq_len, config.hidden_size, 
        dtype=torch.bfloat16, device=device
    )
    
    # 5. Run the block
    print("Running inputs through the batched MoE block...")
    with torch.no_grad(): 
        output, router_logits = moe_block(dummy_input)
        
    # 6. Output Results
    print("\n--- Execution Successful ---")
    print(f"Output shape: {output.shape} | Expected: ({batch_size}, {seq_len}, {config.hidden_size})")
    print(f"Router logits shape: {router_logits.shape} | Expected: ({batch_size * seq_len}, {config.num_local_experts})")
    
    print(f"\nSample output (first token, first 5 dimensions):")
    print(output[0, 0, :5])

if __name__ == "__main__":
    main()