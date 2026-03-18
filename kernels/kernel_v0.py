import numpy as np
import math

import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def routing_kernel(inputs_T, routing_weights): # Deprecated
    """
    MoE routing kernel: tiled matmul followed by softmax.

    Computes probs = softmax(inputs @ routing_weights) for each token.

    Args:
        inputs_T: (feature_dim, T) transposed, flattened token embeddings.
                  Contraction dim (feature_dim) is on the partition axis.
        routing_weights: (feature_dim, num_experts) gating weight matrix.

    Returns:
        probs: (T, num_experts) per-token expert routing probabilities.
    """
    feature_dim, T = inputs_T.shape
    feature_dim_, num_experts = routing_weights.shape

    assert feature_dim_ == feature_dim, f"Feature dim mismatch: {feature_dim_} vs {feature_dim}"

    TILE_K = nl.tile_size.pmax                  # 128
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128

    probs_out = nl.ndarray((T, num_experts), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(T // TILE_M):
        logits = nl.ndarray((TILE_M, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=logits, value=0.0)

        for k in nl.affine_range(feature_dim // TILE_K):
            lhs_tile = nl.ndarray((TILE_K, TILE_M), dtype=inputs_T.dtype, buffer=nl.sbuf)
            rhs_tile = nl.ndarray((TILE_K, num_experts), dtype=routing_weights.dtype, buffer=nl.sbuf)
            partial = nl.ndarray((TILE_M, num_experts), dtype=nl.float32, buffer=nl.psum)

            nisa.dma_copy(dst=lhs_tile,
                          src=inputs_T[k * TILE_K:(k + 1) * TILE_K,
                                       m * TILE_M:(m + 1) * TILE_M])
            nisa.dma_copy(dst=rhs_tile,
                          src=routing_weights[k * TILE_K:(k + 1) * TILE_K,
                                              0:num_experts])
            
            nisa.nc_matmul(partial, lhs_tile, rhs_tile)
            nisa.tensor_tensor(dst=logits, data1=partial, data2=logits, op=nl.add)

        # softmax data alloc
        row_max = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        shifted = nl.ndarray((TILE_M, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        exp_vals = nl.ndarray((TILE_M, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        sum_exp = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        sum_exp_reciprocal = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        probs = nl.ndarray((TILE_M, num_experts), dtype=nl.float32, buffer=nl.sbuf)

        # softmax ops
        nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=logits, axis=1, keepdims=True)
        nisa.tensor_scalar(dst=shifted, data=logits, op0=nl.subtract, operand0=row_max)
        nisa.activation(dst=exp_vals, data=shifted, op=nl.exp)
        nisa.tensor_reduce(dst=sum_exp, data=exp_vals, op=nl.add, axis=1, keepdims=True)
        nisa.reciprocal(dst=sum_exp_reciprocal, data=sum_exp)
        nisa.tensor_scalar(dst=probs, data=exp_vals, op0=nl.multiply, operand0=sum_exp_reciprocal)

        nisa.dma_copy(dst=probs_out[m * TILE_M:(m + 1) * TILE_M, 0:num_experts], src=probs)
       
    return probs_out


@nki.jit
def moe_expert_kernel(sorted_tokens, w1_experts, w3_experts, w2_experts, expert_offsets):
    feature_dim, T = sorted_tokens.shape
    _, expert_dim = w1_experts.shape
    num_experts = expert_offsets.shape[0] - 1
    MAX_TOK_PER_EXPERT = T // num_experts
    
    # NKI GEMM: LHS^T * RHS
    TILE_M = nl.tile_size.gemm_stationary_fmax # stationary dim, (T), tile size = 128
    TILE_K = nl.tile_size.pmax # partition dim, contraction dim (feature_dim), tile size = 128
    TILE_N = nl.tile_size.gemm_moving_fmax # moving dim, (expert/intermediate_dim), tile size = 512

    num_m_tiles = MAX_TOK_PER_EXPERT // TILE_M
    num_k_tiles = feature_dim // TILE_K
    num_n_tiles = expert_dim // TILE_N

    output_tokens = nl.ndarray((T, feature_dim), dtype=sorted_tokens.dtype, buffer=nl.shared_hbm)

    for expert in nl.affine_range(num_experts):
        tok_offset = expert * MAX_TOK_PER_EXPERT
        weight_offset = expert * feature_dim
        weight_offset_d = expert * expert_dim

        for m in nl.affine_range(num_m_tiles):
            # bf16 intermediate: safe to truncate here; the down-projection
            # GEMM accumulates back into fp32.
            
            intermediate = nl.ndarray((TILE_M, expert_dim), dtype=sorted_tokens.dtype, buffer=nl.sbuf)

            # ── Gate / Up projections ──
            for n in nl.affine_range(num_n_tiles):
                gate_acc = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum, name=f"gate_acc_{expert}_{m}_{n}")
                up_acc = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum, name=f"up_acc_{expert}_{m}_{n}")

                for k in nl.affine_range(num_k_tiles):
                    lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=sorted_tokens.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=lhsT_tile, src=sorted_tokens[
                        k * TILE_K : (k + 1) * TILE_K,
                        tok_offset + m * TILE_M : tok_offset + ((m + 1) * TILE_M)
                    ])

                    rhs_gate = nl.ndarray((TILE_K, TILE_N), dtype=w1_experts.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=rhs_gate, src=w1_experts[
                        weight_offset + (k * TILE_K) : weight_offset + ((k + 1) * TILE_K),
                        n * TILE_N : (n + 1) * TILE_N
                    ])

                    nisa.nc_matmul(gate_acc, lhsT_tile, rhs_gate)

                    rhs_up = nl.ndarray((TILE_K, TILE_N), dtype=w3_experts.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=rhs_up, src=w3_experts[
                        weight_offset + (k * TILE_K) : weight_offset + ((k + 1) * TILE_K),
                        n * TILE_N : (n + 1) * TILE_N
                    ])

                    nisa.nc_matmul(up_acc, lhsT_tile, rhs_up)

                # SiLU activation in fp32 (sigmoid is numerically sensitive)
                #sig_gate = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                silu_gate = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=silu_gate, op=nl.silu, data=gate_acc)

                
                #nisa.tensor_tensor(dst=silu_gate, data1=gate_acc, data2=silu_gate, op=nl.multiply)

                # Truncate to bf16 on write to intermediate
                out_tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=out_tile, data1=silu_gate, data2=up_acc, op=nl.multiply)
                nisa.dma_copy(dst=intermediate[:, n * TILE_N : (n + 1) * TILE_N], src=out_tile)

            # ── Down projection ──
            num_nd_tiles = feature_dim // TILE_N
            num_kd_tiles = expert_dim // TILE_K
            for n in nl.affine_range(num_nd_tiles):
                out_acc = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf, name=f"out_acc_{expert}_{m}_{n}")

                for k in nl.affine_range(num_kd_tiles):
                    # load in transposed intermediate tensor (nc_transpose requires 32x32 tiles)
                    lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=intermediate.dtype, buffer=nl.sbuf)
                    for i in nl.affine_range(TILE_M // 32):
                        for j in nl.affine_range(TILE_K // 32):
                            nisa.nc_transpose(
                                dst=lhsT_tile[j * 32 : (j + 1) * 32, i * 32 : (i + 1) * 32],
                                data=intermediate[
                                    i * 32 : (i + 1) * 32,
                                    (k * TILE_K) + (j * 32) : (k * TILE_K) + ((j + 1) * 32)
                                ]
                            )

                    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=w2_experts.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=rhs_tile, src=w2_experts[
                        weight_offset_d + (k * TILE_K) : weight_offset_d + ((k + 1) * TILE_K),
                        n * TILE_N : (n + 1) * TILE_N
                    ])

                    out_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(out_psum, lhsT_tile, rhs_tile)
                    nisa.tensor_tensor(dst=out_acc, data1=out_psum, data2=out_acc, op=nl.add)

                nisa.dma_copy(dst=output_tokens[
                    tok_offset + (m * TILE_M) : tok_offset + ((m + 1) * TILE_M),
                    n * TILE_N : (n + 1) * TILE_N
                ], src=out_acc)

    return output_tokens

