import numpy as np
import math

import nki
import nki.language as nl
import nki.isa as nisa
#import neuronxcc.nki as nki
#import neuronxcc.nki.language as nl
#import neuronxcc.nki.isa as nisa


@nki.jit
def routing_kernel(inputs_T, routing_weights):
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

        nisa.dma_copy(dst=probs_out[m * TILE_M:(m + 1) * TILE_M,
                                    0:num_experts], src=probs)
       
    return probs_out

@nki.jit
def moe_kernel(inputs, expert_w_gate, expert_w_up, expert_w_down,
                   top_k_indices, top_k_values):
    """
    MoE expert computation using explicit nisa-style operations.

    Functionally identical to moe_kernel but rewritten to use the
    nisa.* ISA-level call style (pre-declared buffers, explicit
    dma_copy / tensor_tensor / tensor_scalar / activation / reciprocal)
    instead of the nl.* math helpers.

    Args:
        inputs:         (T, feature_dim) flattened token embeddings
        expert_w_gate:  (num_experts * feature_dim, expert_dim) gate proj
        expert_w_up:    (num_experts * feature_dim, expert_dim) up proj
        expert_w_down:  (num_experts * expert_dim, feature_dim) down proj
        top_k_indices:  (T, top_k) selected expert indices as float32
        top_k_values:   (T, top_k) routing weights

    Returns:
        output: (T, feature_dim)
    """
    T, feature_dim = inputs.shape
    _, expert_dim = expert_w_gate.shape
    _, top_k = top_k_indices.shape

    TILE_K = nl.tile_size.pmax              # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    num_k_feat = feature_dim // TILE_K
    num_n_expert = expert_dim // TILE_N
    num_k_expert = expert_dim // TILE_K
    num_n_feat = feature_dim // TILE_N

    output = nl.ndarray((T, feature_dim), dtype=nl.float32,
                        buffer=nl.shared_hbm)

    # iterate sequentially over tokens
    for t in nl.affine_range(T):
        # --- Pre-load token and transpose to column vectors ---
        x_t = nl.ndarray((nl.par_dim(TILE_K), num_k_feat), dtype=nl.float32, buffer=nl.sbuf)
        for k_load in nl.affine_range(num_k_feat):
            x_row = nl.ndarray((1, TILE_K), dtype=inputs.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_row, src=inputs[t:t+1, k_load * TILE_K:(k_load + 1) * TILE_K])

            # transpose (I should just do this on the host)
            x_col_psum = nl.ndarray((TILE_K, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=x_col_psum, data=x_row)

            nisa.tensor_copy(dst=x_t[:, k_load:k_load+1], src=x_col_psum, dtype=nl.float32)
            #x_t[:, k_load:k_load+1] = nisa.tensor_copy(
            #    x_col_psum, dtype=nl.float32)

        out_token = nl.ndarray((1, feature_dim), dtype=nl.float32,
                               buffer=nl.sbuf)
        nisa.memset(dst=out_token, value=0.0)

        # iterate over experts
        for k_top in nl.static_range(top_k):
            # expert index
            expert_idx = nl.ndarray((1, 1), dtype=top_k_indices.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=expert_idx, src=top_k_indices[t:t+1, k_top:k_top+1])

            # expert weight
            weight = nl.ndarray((1, 1), dtype=top_k_values.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=weight, src=top_k_values[t:t+1, k_top:k_top+1])

            # Row offsets into the flattened 2-D weight tensors
            gate_up_base = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=gate_up_base, data=expert_idx, op0=nl.multiply, operand0=feature_dim)

            down_base = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=down_base, data=expert_idx, op0=nl.multiply, operand0=expert_dim)

            # --- Gate / Up projections with SiLU fusion ---
            intermediate = nl.ndarray((1, expert_dim), dtype=nl.float32, buffer=nl.sbuf)

            for n in nl.affine_range(num_n_expert):
                gate_acc = nl.ndarray((1, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                up_acc = nl.ndarray((1, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=gate_acc, value=0.0)
                nisa.memset(dst=up_acc, value=0.0)

                for k in nl.affine_range(num_k_feat):
                    x_k = x_t[:, k:k+1]

                    row_off = nl.ndarray((1, 1), dtype=nl.float32,
                                         buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=row_off, data=gate_up_base,
                                       op0=nl.add, operand0=k * TILE_K)

                    row_off_reg = nisa.register_alloc()
                    nisa.register_load(row_off_reg, row_off)

                    w_g = nl.ndarray((TILE_K, TILE_N), dtype=expert_w_gate.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=w_g, src=expert_w_gate.ap(
                        pattern=[[expert_dim, TILE_K], [1, TILE_N]],
                        offset=n*TILE_N, 
                        scalar_offset=row_off_reg,
                        indirect_dim=0
                    ))

                    w_u = nl.ndarray((TILE_K, TILE_N), dtype=expert_w_up.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=w_u, src=expert_w_up.ap(
                        pattern=[[expert_dim, TILE_K], [1, TILE_N]],
                        offset=n*TILE_N, 
                        scalar_offset=row_off_reg,
                        indirect_dim=0
                    ))

                    """w_g = nl.load(expert_w_gate.ap(
                        pattern=[[expert_dim, TILE_K], [1, TILE_N]],
                        offset=n * TILE_N,
                        scalar_offset=row_off_reg,
                        indirect_dim=0))
                    w_u = nl.load(expert_w_up.ap(
                        pattern=[[expert_dim, TILE_K], [1, TILE_N]],
                        offset=n * TILE_N,
                        scalar_offset=row_off_reg,
                        indirect_dim=0))"""

                    gate_partial = nl.ndarray((1, TILE_N), dtype=nl.float32,
                                              buffer=nl.psum)
                    up_partial = nl.ndarray((1, TILE_N), dtype=nl.float32,
                                            buffer=nl.psum)

                    nisa.nc_matmul(gate_partial, x_k, w_g)
                    nisa.nc_matmul(up_partial, x_k, w_u)

                    nisa.tensor_tensor(dst=gate_acc, data1=gate_partial,
                                       data2=gate_acc, op=nl.add)
                    nisa.tensor_tensor(dst=up_acc, data1=up_partial,
                                       data2=up_acc, op=nl.add)

                # SiLU(gate) * up
                neg_gate = nl.ndarray((1, TILE_N), dtype=nl.float32,
                                      buffer=nl.sbuf)
                nisa.tensor_scalar(dst=neg_gate, data=gate_acc,
                                   op0=nl.multiply, operand0=-1.0)

                exp_neg = nl.ndarray((1, TILE_N), dtype=nl.float32,
                                     buffer=nl.sbuf)
                nisa.activation(dst=exp_neg, data=neg_gate, op=nl.exp)

                denom = nl.ndarray((1, TILE_N), dtype=nl.float32,
                                   buffer=nl.sbuf)
                nisa.tensor_scalar(dst=denom, data=exp_neg,
                                   op0=nl.add, operand0=1.0)

                inv_denom = nl.ndarray((1, TILE_N), dtype=nl.float32,
                                       buffer=nl.sbuf)
                nisa.reciprocal(dst=inv_denom, data=denom)

                silu = nl.ndarray((1, TILE_N), dtype=nl.float32,
                                  buffer=nl.sbuf)
                nisa.tensor_tensor(dst=silu, data1=gate_acc,
                                   data2=inv_denom, op=nl.multiply)

                #inter_tile = nl.ndarray((1, TILE_N), dtype=nl.float32,
                #                        buffer=nl.sbuf)

                nisa.tensor_tensor(dst=intermediate[:, nl.ds(n * TILE_N, TILE_N)], data1=silu,
                                   data2=up_acc, op=nl.multiply)

                #intermediate[:, nl.ds(n * TILE_N, TILE_N)] = inter_tile

            # --- Down projection ---
            down_result = nl.ndarray((1, feature_dim), dtype=nl.float32,
                                     buffer=nl.sbuf)

            for n_d in nl.affine_range(num_n_feat):
                down_acc = nl.ndarray((1, TILE_N), dtype=nl.float32,
                                      buffer=nl.sbuf)
                nisa.memset(dst=down_acc, value=0.0)

                for k_d in nl.affine_range(num_k_expert):
                    inter_slice = intermediate[:, nl.ds(k_d * TILE_K, TILE_K)]

                    #inter_t_psum = nisa.nc_transpose(inter_slice)
                    inter_t_psum = nl.ndarray((TILE_K, 1), dtype=intermediate.dtype, buffer=nl.psum)
                    inter_t_sbuf = nl.ndarray((TILE_K, 1), dtype=intermediate.dtype, buffer=nl.sbuf)
                    nisa.nc_transpose(dst=inter_t_psum, data=inter_slice)
                    nisa.tensor_copy(dst=inter_t_sbuf, src=inter_t_psum)

                    row_off_d = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=row_off_d, data=down_base, op0=nl.add, operand0=k_d * TILE_K)

                    row_off_d_reg = nisa.register_alloc()
                    nisa.register_load(row_off_d_reg, row_off_d)

                    w_d = nl.ndarray((TILE_K, TILE_N), dtype=expert_w_down.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=w_d, src=expert_w_down.ap(
                        pattern=[[feature_dim, TILE_K], [1, TILE_N]],
                        offset=n_d*TILE_N,
                        scalar_offset=row_off_d_reg,
                        indirect_dim=0
                    ))
                    """w_d = nl.load(expert_w_down.ap(
                        pattern=[[feature_dim, TILE_K], [1, TILE_N]],
                        offset=n_d * TILE_N,
                        scalar_offset=row_off_d_reg,
                        indirect_dim=0))"""

                    down_partial = nl.ndarray((1, TILE_N), dtype=nl.float32,
                                              buffer=nl.psum)
                    nisa.nc_matmul(down_partial, inter_t_sbuf, w_d)

                    nisa.tensor_tensor(dst=down_acc, data1=down_partial,
                                       data2=down_acc, op=nl.add)

                #down_result[:, nl.ds(n_d * TILE_N, TILE_N)] = down_acc
                nisa.tensor_copy(dst=down_result[:, nl.ds(n_d * TILE_N, TILE_N)], src=down_acc)

            # Weight by routing score and accumulate
            weighted = nl.ndarray((1, feature_dim), dtype=nl.float32,
                                  buffer=nl.sbuf)
            nisa.tensor_scalar(dst=weighted, data=down_result,
                               op0=nl.multiply, operand0=weight)
            nisa.tensor_tensor(dst=out_token, data1=out_token,
                               data2=weighted, op=nl.add)

        nisa.dma_copy(dst=output[t:t+1, 0:feature_dim], src=out_token)

    return output



@nki.jit
def old_moe_kernel(inputs, expert_w_gate, expert_w_up, expert_w_down,
               top_k_indices, top_k_values):
    """
    Naive MoE expert computation — single-token sequential baseline.

    Iterates over every token individually. For each token, looks up
    the top-k expert indices, runs the SwiGLU MLP (gate/up/down) for
    each selected expert, and accumulates the weighted results.

    Weight tensors are passed pre-flattened (expert dim folded into rows)
    so that nl.ds can dynamically select the correct expert at runtime.

    Args:
        inputs:         (T, feature_dim) flattened token embeddings
        expert_w_gate:  (num_experts * feature_dim, expert_dim) gate proj
        expert_w_up:    (num_experts * feature_dim, expert_dim) up proj
        expert_w_down:  (num_experts * expert_dim, feature_dim) down proj
        top_k_indices:  (T, top_k) selected expert indices as float32
        top_k_values:   (T, top_k) routing weights

    Returns:
        output: (T, feature_dim)
    """
    T, feature_dim = inputs.shape
    _, expert_dim = expert_w_gate.shape
    _, top_k = top_k_indices.shape

    TILE_K = nl.tile_size.pmax              # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    num_k_feat = feature_dim // TILE_K # 
    num_n_expert = expert_dim // TILE_N # output dim for up/gate
    num_k_expert = expert_dim // TILE_K
    num_n_feat = feature_dim // TILE_N # output dim for down

    output = nl.ndarray((T, feature_dim), dtype=nl.float32,
                        buffer=nl.shared_hbm)

    for t in nl.sequential_range(T):
        # --- Pre-load token and transpose to column vectors ---
        # Each column k of x_t holds inputs[t, k*128:(k+1)*128] transposed
        # to (TILE_K, 1) so it can serve as nc_matmul stationary operand.
        # x_t = single token, tiled
        x_t = nl.ndarray((nl.par_dim(TILE_K), num_k_feat),
                         dtype=nl.float32, buffer=nl.sbuf)
        for k_ld in nl.affine_range(num_k_feat):
            x_row = nl.load(
                inputs[t:t+1, k_ld * TILE_K:(k_ld + 1) * TILE_K])
            x_col_psum = nisa.nc_transpose(x_row)
            x_t[:, k_ld:k_ld+1] = nisa.tensor_copy(
                x_col_psum, dtype=nl.float32)
            # x_t shape = 128, 32. feature vector for a token in 
        out_token = nl.zeros((1, feature_dim), dtype=nl.float32,
                             buffer=nl.sbuf)

        for k_top in nl.static_range(top_k):
            #expert_idx = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            #nisa.dma_copy(dst=expert_idx, src=top_k_indices[t:t+1, k_top:k_top+1])
            expert_idx = nl.load(
                top_k_indices[t:t+1, k_top:k_top+1])
            weight = nl.load(
                top_k_values[t:t+1, k_top:k_top+1])

            # Row offsets into the flattened 2-D weight tensors
            gate_up_base = nl.multiply(expert_idx, feature_dim)
            down_base = nl.multiply(expert_idx, expert_dim)

            # Tiled GEMM operations for gate and up projection, fused with SiLU
            intermediate = nl.ndarray((1, expert_dim),
                                      dtype=nl.float32, buffer=nl.sbuf)
            for n in nl.affine_range(num_n_expert):
                gate_psum = nl.zeros((1, TILE_N), dtype=nl.float32,
                                     buffer=nl.psum)
                up_psum = nl.zeros((1, TILE_N), dtype=nl.float32,
                                   buffer=nl.psum)

                for k in nl.affine_range(num_k_feat):
                    x_k = x_t[:, k:k+1]                       # (128, 1)
                    row_off = gate_up_base + k * TILE_K

                    row_off_reg = nisa.register_alloc()
                    nisa.register_load(row_off_reg, row_off)

                    w_g = nl.load(expert_w_gate.ap(
                        pattern=[[expert_dim, TILE_K], [1, TILE_N]],
                        offset=n * TILE_N,
                        scalar_offset=row_off_reg,
                        indirect_dim=0
                    ))
                    w_u = nl.load(expert_w_up.ap(
                        pattern=[[expert_dim, TILE_K], [1, TILE_N]],
                        offset=n * TILE_N,
                        scalar_offset=row_off_reg,
                        indirect_dim=0
                    ))

                    """w_g = nl.load(expert_w_gate[
                        nl.ds(row_off[0, 0], TILE_K),
                        n * TILE_N:(n + 1) * TILE_N])
                    w_u = nl.load(expert_w_up[
                        nl.ds(row_off[0, 0], TILE_K),
                        n * TILE_N:(n + 1) * TILE_N])"""

                    gate_psum += nisa.nc_matmul(x_k, w_g)
                    up_psum += nisa.nc_matmul(x_k, w_u)

                gate_s = nisa.tensor_copy(gate_psum, dtype=nl.float32)
                up_s = nisa.tensor_copy(up_psum, dtype=nl.float32)

                # SiLU(x) = x / (1 + exp(-x))
                neg_gate = nl.subtract(0.0, gate_s)
                silu = nl.divide(gate_s,
                                 nl.add(nl.exp(neg_gate), 1.0))
                inter_tile = nl.multiply(silu, up_s)

                intermediate[:, nl.ds(n * TILE_N, TILE_N)] = inter_tile

            # ---- Down projection ----
            down_result = nl.ndarray((1, feature_dim),
                                     dtype=nl.float32, buffer=nl.sbuf)

            for n_d in nl.affine_range(num_n_feat):
                down_psum = nl.zeros((1, TILE_N), dtype=nl.float32,
                                     buffer=nl.psum)

                for k_d in nl.affine_range(num_k_expert):
                    # Transpose intermediate chunk for contraction
                    inter_slice = intermediate[
                        :, nl.ds(k_d * TILE_K, TILE_K)]       # (1, 128)
                    inter_t_psum = nisa.nc_transpose(inter_slice)
                    inter_t = nisa.tensor_copy(
                        inter_t_psum, dtype=nl.float32)        # (128, 1)

                    row_off_d = down_base + k_d * TILE_K
                    row_off_d_reg = nisa.register_alloc()
                    nisa.register_load(row_off_d_reg, row_off_d)
                    
                    w_d = nl.load(expert_w_down.ap(
                        pattern=[[feature_dim, TILE_K], [1, TILE_N]],
                        offset=n_d * TILE_N,
                        scalar_offset=row_off_d_reg,
                        indirect_dim=0
                    ))
                    """
                    w_d = nl.load(expert_w_down[
                        nl.ds(row_off_d[0, 0], TILE_K),
                        n_d * TILE_N:(n_d + 1) * TILE_N])"""

                    down_psum += nisa.nc_matmul(inter_t, w_d)

                down_tile = nisa.tensor_copy(
                    down_psum, dtype=nl.float32)
                down_result[
                    :, nl.ds(n_d * TILE_N, TILE_N)] = down_tile

            weighted = nl.multiply(down_result, weight)
            out_token = nl.add(out_token, weighted)

        nl.store(dst=output[t:t+1, 0:feature_dim], value=out_token)

    return output
