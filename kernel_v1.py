import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import nki
import nki.language as nl
import nki.isa as nisa

import torch
from torch_xla.core import xla_model as xm

from mistral_moe import MixtralConfig, MixtralSparseMoeBlock, MixtralRouter, MixtralExpert

MODEL_PATH = "weights/uncompiled_model_weights/mistral_moe.pt"
INPUT_TENSOR_PATH = "weights/input_weights/prefill_b1_s128.pt"
MAX_TOKENS_PER_EXPERT = 128 # TODO: eventually get rid of this or expand a bunch



# TODO: make absolutely sure this works
# creates the properly formatted input for the kernel
# input_tensor: batch_size x sequence_length x embedding_dim
def create_inputs(input_tensor):
    # get dimensions and reshape input
    batch_size, sequence_length, hidden_dim = input_tensor.shape
    input_tensor = input_tensor.view(-1, hidden_dim)

    # create moe block
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    moe_block = MixtralSparseMoeBlock(MixtralConfig())
    moe_block.load_state_dict(state_dict)
    moe_block = moe_block.to(dtype=torch.bfloat16)
    moe_block.eval()

    # get router outputs (of size batch_size * sequence_length x top_k)
    with torch.no_grad():
        routing_weights, selected_experts = moe_block.router(input_tensor)

    N, top_k = routing_weights.shape

    # 1. Flatten router outputs
    flat_routing_weights = routing_weights.view(-1)
    flat_selected_experts = selected_experts.view(-1)

    # 2. Duplicate input tokens for top-k routing
    repeated_tokens = input_tensor.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)

    # 3. Sort tokens and weights by assigned expert
    sorted_expert_indices = torch.argsort(flat_selected_experts)
    sorted_tokens = repeated_tokens[sorted_expert_indices].to(dtype=torch.bfloat16)
    sorted_routing_weights = flat_routing_weights[sorted_expert_indices].to(dtype=torch.bfloat16)
    sorted_experts = flat_selected_experts[sorted_expert_indices]

    # 4. Calculate token counts per expert
    num_experts = moe_block.num_experts
    expert_counts = torch.bincount(sorted_experts, minlength=num_experts)

    # 5. Calculate padded counts (multiple of 128 for NKI tile size)
    TILE_SIZE = 128
    padded_expert_counts = torch.ceil(expert_counts / TILE_SIZE).to(torch.int32) * TILE_SIZE
    total_padded_tokens = padded_expert_counts.sum().item()

    # 6. Pre-allocate padded tensors
    padded_sorted_tokens = torch.zeros((total_padded_tokens, hidden_dim), dtype=torch.bfloat16)
    padded_routing_weights = torch.zeros(total_padded_tokens, dtype=torch.bfloat16)
    
    # Calculate new offsets based on padded counts
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32)
    expert_offsets[1:] = torch.cumsum(padded_expert_counts, dim=0)

    # 7. Copy the real tokens into the padded buffers
    current_read_idx = 0
    current_write_idx = 0
    
    for i in range(num_experts):
        count = expert_counts[i].item()
        padded_count = padded_expert_counts[i].item()
        
        if count > 0:
            padded_sorted_tokens[current_write_idx : current_write_idx + count] = \
                sorted_tokens[current_read_idx : current_read_idx + count]
            padded_routing_weights[current_write_idx : current_write_idx + count] = \
                sorted_routing_weights[current_read_idx : current_read_idx + count]
            
        current_read_idx += count
        current_write_idx += padded_count

    # 8. Extract and transpose expert weights (PyTorch [out, in] -> NKI [in, out])
    w1_list, w2_list, w3_list = [], [], []
    for expert in moe_block.experts:
        w1_list.append(expert.w1.weight.T)
        w2_list.append(expert.w2.weight.T)
        w3_list.append(expert.w3.weight.T)

    w1_experts = torch.stack(w1_list).to(dtype=torch.bfloat16).contiguous()
    w3_experts = torch.stack(w3_list).to(dtype=torch.bfloat16).contiguous()
    w2_experts = torch.stack(w2_list).to(dtype=torch.bfloat16).contiguous()

    # 9. Pre-allocate the output buffer sized to the padded T
    output_tokens = torch.zeros((total_padded_tokens, hidden_dim), dtype=torch.bfloat16)

    return (
        (padded_sorted_tokens.T).contiguous(),
        w1_experts,
        w3_experts,
        w2_experts,
        padded_routing_weights.contiguous(),
        expert_offsets
    )



'''
Notes:
    - everything is bf16
    - T = batch_size * sequence_length * topk = 16 * 4096 * 2 = 131072
    - output should be T x 4096
'''
@nki.jit
def fused_expert_kernel(
    sorted_tokens,     # [4096, T]          - The grouped input tokens transposed
    w1_experts,        # [8, 4096, 14336]   - All 8 Gate weight matrices
    w3_experts,        # [8, 4096, 14336]   - All 8 Up weight matrices
    w2_experts,        # [8, 14336, 4096]   - All 8 Down weight matrices
    routing_weights    # [T]                - The scalar probability per token
    ):

    # get shapes
    feature_dim, T = sorted_tokens.shape
    num_experts, _, intermediate_dim = w1_experts.shape
    
    # assert shapes
    assert feature_dim == w1_experts.shape[1], "feature_dim must match w1_experts.shape[1]"
    assert intermediate_dim == w2_experts.shape[1], "intermediate_dim must match w2_experts.shape[1]"

    # create output tensor
    output = nl.ndarray((T, intermediate_dim), dtype=nl.bfloat16, buffer=nl.hbm)
    
    # loop over experts
    for expert_idx in nl.affine_range(num_experts):

        # grab input tensor and expert matrices
        # NOte: these arent actually used to fetch any data, they are just helpful references
        lhsT = sorted_tokens[:, expert_idx * MAX_TOKENS_PER_EXPERT:(expert_idx + 1) * MAX_TOKENS_PER_EXPERT]
        rhs1 = w1_experts[expert_idx]
        rhs2 = w3_experts[expert_idx]

        # Verify that the lhsT and rhs have the same contraction dimension.
        # K = feature_dim = 4096
        # M = # of tokens routed to current expert
        # N = intermediate_dim = 14336
        # rhs1 is W1
        # rhs2 is W3
        K, M = lhsT.shape
        K_, N = rhs1.shape
        assert K == K_, "lhsT and rhs must have the same contraction dimension"

        # Lookup the device matrix multiply dimensions.
        TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
        TILE_K = nl.tile_size.pmax  # 128
        TILE_N = nl.tile_size.gemm_moving_fmax  # 512

        # Verify that the input matrices are a multiple of the tile dimensions.
        assert M % TILE_M == 0, f"Expected M, {M}, to be a multiple of stationary free-dimension max, {TILE_M}"
        assert N % TILE_N == 0, f"Expected N, {N}, to be a multiple of moving free-dimension max, {TILE_N}"
        assert K % TILE_K == 0, f"Expected K, {K}, to be a multiple of the partition dimension max, {TILE_K}"

        # Use affine_range to loop over tiles
        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):

                # Allocate a tensor in PSUM for both the w1 and w3 matmul results
                res_psum1 = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum, name=f"res_psum1_e_{expert_idx}_m{m}_n{n}") # stores a tile of xW1
                res_psum2 = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum, name=f"res_psum2_e_{expert_idx}_m{m}_n{n}") # stores a tile of xW3

                for k in nl.affine_range(K // TILE_K):
                    # Declare the tiles on SBUF
                    lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                    rhs1_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs1.dtype, buffer=nl.sbuf)
                    rhs2_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs2.dtype, buffer=nl.sbuf)

                    # Load tiles from lhsT and rhs1/2
                    # need to slice directly
                    expert_start = expert_idx * MAX_TOKENS_PER_EXPERT

                    nisa.dma_copy(dst=lhsT_tile,
                                  src=sorted_tokens[k * TILE_K:(k + 1) * TILE_K,
                                                    expert_start + (m * TILE_M): expert_start + (m + 1) * TILE_M])

                    nisa.dma_copy(dst=rhs1_tile,
                                  src=w1_experts[expert_idx, 
                                                 k * TILE_K:(k + 1) * TILE_K,
                                                 n * TILE_N:(n + 1) * TILE_N])

                    nisa.dma_copy(dst=rhs2_tile,
                                  src=w3_experts[expert_idx, 
                                                 k * TILE_K:(k + 1) * TILE_K,
                                                 n * TILE_N:(n + 1) * TILE_N])

                    # Accumulate partial-sums into PSUM
                    # note that this use of nc_matmul does actually accumulate into the tile
                    nisa.nc_matmul(dst=res_psum1, stationary=lhsT_tile, moving=rhs1_tile)
                    nisa.nc_matmul(dst=res_psum2, stationary=lhsT_tile, moving=rhs2_tile)

                # Apply silu activation to res_psum1 to compute swish(xW1) tile in sbuf
                # TODO: does this automatically get cast down?
                swish_sbuf_tile = nl.ndarray((TILE_M, TILE_N), dtype=output.dtype, buffer=nl.sbuf) # stores a tile of swish(xW1)
                nisa.activation(dst=swish_sbuf_tile, data=res_psum1, op=nl.silu)

                # Allocate sbuf array to hold final tile result then apply hamarand and copy out to hbm
                # stores swish(xW1) * xW3 tile
                final_output_tile_sbuf = nl.ndarray(swish_sbuf_tile.shape, dtype=output.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=final_output_tile_sbuf, data1=swish_sbuf_tile, data2=res_psum2, op=nl.multiply)

                # Copy the result from SBUF to HBM
                # need to ajust for the fact that the output tensor is T x N, but we are computing M x N tiles
                start = MAX_TOKENS_PER_EXPERT * expert_idx + m * TILE_M
                end = start + TILE_M
                nisa.dma_copy(dst=output[start:end, n * TILE_N:(n + 1) * TILE_N], src=final_output_tile_sbuf)

    return output





def test():
    # load in the input tensor

    input_tensor = torch.load(INPUT_TENSOR_PATH)
    input_tensor = input_tensor.to(dtype=torch.bfloat16, device="cpu")

    # create inputs for the kernel
    device = xm.xla_device()
    sorted_tokens, w1_experts, w3_experts, w2_experts, routing_weights, expert_offsets = create_inputs(input_tensor)
    # print(f"sorted_tokens shape: {sorted_tokens.shape}")
    # print(f"w1_experts shape: {w1_experts.shape}")
    # print(f"w3_experts shape: {w3_experts.shape}")
    # print(f"w2_experts shape: {w2_experts.shape}")
    # print(f"routing_weights shape: {routing_weights.shape}")
    # print(f"expert_offsets shape: {expert_offsets.shape}")
    # print(f"expert last = {expert_offsets[-1]}")
    # for i in range(expert_offsets.shape[0]-1):
    #     print(f"expert #{i} entries= {expert_offsets[i]} to {expert_offsets[i+1]}")

    # Run NKI kernel
    output_small = fused_expert_kernel(sorted_tokens.to(device), w1_experts.to(device), w3_experts.to(device), w2_experts.to(device), routing_weights.to(device))
    output_small = output_small.cpu()
    print(output_small.shape)
    print(output_small)

    # # Run torch reference
    # output_small_torch = ...

    # # Compare results
    # print("Checking correctness of nki_matmul_basic")
    # if torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2):
    # print("NKI and Torch match")
    # else:
    # print("NKI and Torch differ")

if __name__ == "__main__":
    test()
