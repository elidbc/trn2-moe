import numpy as np
import math

import nki
import nki.language as nl
import nki.isa as nisa



# MoE layer kernel - improves on v0 by fusing the up with the down and transpose projection. 
# Still uses pre-sorted tokens

@nki.jit
def moe_expert_kernel_v2(sorted_tokens, w1_experts, w3_experts, w2_experts, expert_offsets):
    feature_dim, T = sorted_tokens.shape
    _, expert_dim = w1_experts.shape
    num_experts = expert_offsets.shape[0] - 1
    MAX_TOK_PER_EXPERT = T // num_experts

    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax
    NUM_BLOCKS = TILE_N // TILE_M

    num_mup_tiles = expert_dim // TILE_M
    num_kup_tiles = feature_dim // TILE_K
    num_nup_tiles = MAX_TOK_PER_EXPERT // TILE_N

    #num_mdown_tiles = MAX_TOK_PER_EXPERT // TILE_M
    #num_kdown_tiles = expert_dim // TILE_K
    num_ndown_tiles = feature_dim // TILE_N

    output_tokens = nl.ndarray((T, feature_dim), dtype=sorted_tokens.dtype, buffer=nl.shared_hbm)

    for expert in nl.affine_range(num_experts):
        tok_offset = expert * MAX_TOK_PER_EXPERT
        expert_start = expert * feature_dim
        expert_start_down = expert * expert_dim

        for n_up in nl.affine_range(num_nup_tiles):                     # n_up is now token_dim
            
            for n_down in nl.affine_range(num_ndown_tiles):             # n_down is now expert_dim 
                out_psums = [
                    nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum, name=f"out_acc_{expert}_{n_up}_{n_down}_{0}"),
                    nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum, name=f"out_acc_{expert}_{n_up}_{n_down}_{1}"),
                    nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum, name=f"out_acc_{expert}_{n_up}_{n_down}_{2}"),
                    nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum, name=f"out_acc_{expert}_{n_up}_{n_down}_{3}")
                ]
                for m_up in nl.affine_range(num_mup_tiles):         # m_up is now expert_dim 
                    gate_acc = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum, name=f"gate_acc_{expert}_{n_up}_{n_down}_{m_up}")
                    up_acc = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum, name=f"up_acc_{expert}_{n_up}_{n_down}_{m_up}")

                    for k_up in nl.affine_range(num_kup_tiles):     # k_up is now feature_dim 
                        w1_tile = nl.ndarray((TILE_K, TILE_M), dtype=w1_experts.dtype, buffer=nl.sbuf)
                        w3_tile = nl.ndarray((TILE_K, TILE_M), dtype=w3_experts.dtype, buffer=nl.sbuf)
                        tok_tile = nl.ndarray((TILE_K, TILE_N), dtype=sorted_tokens.dtype, buffer=nl.sbuf)

                        nisa.dma_copy(dst=w1_tile, src=w1_experts[
                            expert_start + k_up * TILE_K : expert_start + (k_up + 1) * TILE_K, 
                            m_up * TILE_M : (m_up + 1) * TILE_M
                        ])
                        nisa.dma_copy(dst=w3_tile, src=w3_experts[
                            expert_start + k_up * TILE_K : expert_start + (k_up + 1) * TILE_K, 
                            m_up * TILE_M : (m_up + 1) * TILE_M
                        ])

                        nisa.dma_copy(dst=tok_tile, src=sorted_tokens[
                            k_up * TILE_K : (k_up + 1) * TILE_K,
                            tok_offset + n_up * TILE_N : tok_offset + ((n_up + 1) * TILE_N)
                        ])

                        nisa.nc_matmul(gate_acc, w1_tile, tok_tile)
                        nisa.nc_matmul(up_acc, w3_tile, tok_tile)
                    
                    # ------ SILU activation -------
                    gate_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                    up_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                    silu_gate = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                    temp = nl.ndarray((TILE_M, TILE_N), dtype=sorted_tokens.dtype, buffer=nl.sbuf)

                    nisa.tensor_copy(dst=gate_sbuf, src=gate_acc, dtype=nl.float32)
                    nisa.tensor_copy(dst=up_sbuf, src=up_acc, dtype=nl.float32)
                    nisa.ativation(dst=silu_gate, op=nl.silu, data=gate_sbuf)
                    nisa.tensor_tensor(dst=temp, data1=silu_gate, data2=up_sbuf, op=nl.multiply)

                    # ------ DOWN PHASE ------
                    
                    w2_tile = nl.ndarray((TILE_K, TILE_N), dtype=w2_experts.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=w2_tile, src=w2_experts[
                        expert_start_down + m_up * TILE_M : expert_start_down + ((m_up + 1) * TILE_M),
                        n_down * TILE_N : (n_down + 1) * TILE_N
                    ])

                    for block in nl.affine_range(4):
                        temp_block = nl.ndarray((TILE_M, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=temp_block, src=temp[:, block * TILE_M : (block + 1) * TILE_M])
                        nisa.nc_matmul(out_psums[block], temp_block, w2_tile)
                
                for block in nl.affine_range(4):
                    out_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=sorted_tokens.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=out_sbuf, src=out_psums[block], dtype=sorted_tokens.dtype)
                    nisa.dma_copy(dst=output_tokens[
                        tok_offset + (n_up * TILE_N) + (block * TILE_M) : tok_offset + (n_up * TILE_N) + ((block + 1) * TILE_M),
                        n_down * TILE_N : (n_down + 1) * TILE_N
                    ], src=out_sbuf)

    return output_tokens



                
          