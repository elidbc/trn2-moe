import numpy as np
import math

import nki
import nki.language as nl
import nki.isa as nisa



# MoE layer kernel - improves on v0 by fusing the up with the down and transpose projection. 
# Still uses pre-sorted tokens

@nki.jit
def moe_expert_kernel_v1(sorted_tokens, w1_experts, w3_experts, w2_experts, expert_offsets):
    feature_dim, T = sorted_tokens.shape
    _, expert_dim = w1_experts.shape
    num_experts = expert_offsets.shape[0] - 1
    MAX_TOK_PER_EXPERT = T // num_experts
    
    # NKI GEMM: LHS^T * RHS
    TILE_M = nl.tile_size.gemm_stationary_fmax # stationary dim, (T), tile size = 128
    TILE_K = nl.tile_size.pmax # partition dim (feature_dim otw up, expert_dim otw down), tile size = 128
    TILE_N = nl.tile_size.gemm_moving_fmax # moving dim, output feature_dim, tile size = 512

    num_token_tiles = MAX_TOK_PER_EXPERT // TILE_M # tiles over T

    num_k_tiles = feature_dim // TILE_K # tiles over feature_dim on way up, expert_dim on way down
    num_kd_tiles = expert_dim // TILE_K

    num_out_tiles = feature_dim // TILE_N # tile over feature_dim, 
    

    output_tokens = nl.ndarray((T, feature_dim), dtype=sorted_tokens.dtype, buffer=nl.shared_hbm)

    for expert in nl.affine_range(num_experts):
        tok_offset = expert * MAX_TOK_PER_EXPERT
        weight_offset = expert * feature_dim
        weight_offset_d = expert * expert_dim

        # Tile over token dim
        for m in nl.affine_range(num_token_tiles):
            tok_start = tok_offset + (m * TILE_M)
            tok_end = tok_start + TILE_M
            #out_tile = nl.ndarray((TILE_M, feature_dim), dtype=sorted_tokens.dtype, buffer=nl.sbuf)
            for n in nl.affine_range(num_out_tiles):
                out_acc = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf, name=f"out_acc_{expert}_{m}_{n}")

                for kd in nl.affine_range(num_kd_tiles):
                    h_start = kd * TILE_K
                    h_end = h_start + TILE_K

                    gate_psum = nl.zeros((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.psum, name=f"gate_psum_{expert}_{m}_{n}_{kd}")
                    up_psum = nl.zeros((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.psum, name=f"gate_acc_{expert}_{m}_{n}_{kd}")

                    for k in nl.affine_range(num_k_tiles):
                        x_start = k * TILE_K
                        x_end = x_start + TILE_K

                        tok_tile = nl.ndarray((TILE_K, TILE_M), dtype=sorted_tokens.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(dst=tok_tile, src=sorted_tokens[x_start:x_end, tok_start:tok_end])

                        w1_tile = nl.ndarray((TILE_K, TILE_K), dtype=w1_experts.dtype, buffer=nl.sbuf)
                        w3_tile = nl.ndarray((TILE_K, TILE_K), dtype=w3_experts.dtype, buffer=nl.sbuf)

                        nisa.dma_copy(dst=w1_tile, src=w1_experts[weight_offset + x_start : weight_offset + x_end, h_start : h_end])
                        nisa.dma_copy(dst=w3_tile, src=w3_experts[weight_offset + x_start : weight_offset + x_end, h_start : h_end])

                        nisa.nc_matmul(gate_psum, w1_tile, tok_tile)
                        nisa.nc_matmul(up_psum, w3_tile, tok_tile)

                    gate_sbuf = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
                    up_sbuf = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=gate_sbuf, src=gate_psum, dtype=nl.float32)
                    nisa.tensor_copy(dst=up_sbuf, src=up_psum, dtype=nl.float32)

                    sig_gate = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
                    silu_gate = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
                    temp_t = nl.ndarray((TILE_K, TILE_M), dtype=sorted_tokens.dtype, buffer=nl.sbuf)
                    # make token tile bigger -> 512, 
                    nisa.activation(dst=sig_gate, op=nl.sigmoid, data=gate_sbuf) # sigmoid(gate)
                    nisa.tensor_tensor(dst=silu_gate, data1=gate_sbuf, data2=sig_gate, op=nl.multiply) # gate * sigmoid(gate)
                    nisa.tensor_tensor(dst=temp_t, data1=silu_gate, data2=up_sbuf, op=nl.multiply) # up * (gate * sigmoid(gate))

                    # fused down projection
                    w2_tile = nl.ndarray((TILE_K, TILE_N), dtype=w2_experts.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=w2_tile, src=w2_experts[
                        weight_offset_d + h_start : weight_offset_d + h_end,
                        n * TILE_N : (n + 1) * TILE_N
                    ])

                    partial_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(partial_psum, temp_t, w2_tile)
                    nisa.tensor_tensor(dst=out_acc, data1=partial_psum, data2=out_acc, op=nl.add)
                out_bf16 = nl.ndarray((TILE_M, TILE_N), dtype=sorted_tokens.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=out_bf16, src=out_acc, dtype=sorted_tokens.dtype)
                nisa.dma_copy(dst=output_tokens[tok_start : tok_end, n * TILE_N : (n + 1) * TILE_N], src=out_bf16)

            
    return output_tokens