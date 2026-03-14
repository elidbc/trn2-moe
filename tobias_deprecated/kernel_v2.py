import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import nki
import nki.language as nl
import nki.isa as nisa

import torch
from torch_xla.core import xla_model as xm

from mistral_moe import MixtralConfig, MixtralSparseMoeBlock, MixtralRouter, MixtralExpert

INPUT_TENSOR_PATH = "weights/input_weights/prefill_b1_s128.pt"
MAX_TOKENS_PER_EXPERT = 128 # TODO: eventually get rid of this or expand a bunch



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
        rhs = sorted_tokens[:, expert_idx * MAX_TOKENS_PER_EXPERT:(expert_idx + 1) * MAX_TOKENS_PER_EXPERT]
        lhsT1 = w1_experts[expert_idx]
        lhsT2 = w3_experts[expert_idx]

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

    # Run NKI kernel
    output_small = fused_expert_kernel(sorted_tokens.to(device), w1_experts.to(device), w3_experts.to(device), w2_experts.to(device), routing_weights.to(device))
    output_small = output_small.cpu()
    print(output_small.shape)
    print(output_small)

if __name__ == "__main__":
    test()
