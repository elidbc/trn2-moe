import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


@nki.jit
def matrix_vector_mul_kernel(matT, vec2d):
    """
    Matrix-vector multiplication on Trainium using NKI.

    Computes result = matT^T @ vec2d, i.e. a standard matrix-vector product.

    Args:
        matT:  (K, M) matrix transposed so contraction axis K is on P-dim.
        vec2d: (K, 1) vector reshaped to 2D with K on P-dim.

    Returns:
        result: (M, 1) output in shared HBM.
    """
    K, M = matT.shape
    K_, one = vec2d.shape
    assert K == K_, f"Contraction dim mismatch: {K} vs {K_}"

    result = nl.ndarray((M, one), dtype=matT.dtype, buffer=nl.shared_hbm)

    i_matT_p, i_matT_f = nl.mgrid[0:K, 0:M]
    i_vec_p, i_vec_f = nl.mgrid[0:K, 0:one]
    i_out_p, i_out_f = nl.mgrid[0:M, 0:one]

    matT_tile = nl.load(matT[i_matT_p, i_matT_f])
    vec_tile = nl.load(vec2d[i_vec_p, i_vec_f])

    result_psum = nl.matmul(matT_tile, vec_tile, transpose_x=True)

    result_sbuf = nl.copy(result_psum, dtype=result.dtype)
    nl.store(result[i_out_p, i_out_f], value=result_sbuf)

    return result

@nki.jit
def gemm_nki_kernel(lhsT, rhs):
    """
    Single-tile matrix multiplication on Trainium using NKI.
    Computes C(M,N) = lhsT^T @ rhs where lhsT is (K,M) and rhs is (K,N).
    """
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, f"Contraction dimension mismatch: {K} vs {K_}"
    assert K == 128, f"expected K to be 128, got {K}"
    assert M == 64, f"expected M to be 64, got {M}"
    assert N == 512, f"expected N to be 512, got {N}"

    # final result in shared HBM
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    i_lhsT_p, i_lhsT_f = nl.mgrid[0:K, 0:M]
    i_rhs_p, i_rhs_f = nl.mgrid[0:K, 0:N]
    i_out_p, i_out_f = nl.mgrid[0:M, 0:N]

    lhsT_tile = nl.load(lhsT[i_lhsT_p, i_lhsT_f])
    rhs_tile = nl.load(rhs[i_rhs_p, i_rhs_f])

    result_psum = nl.matmul(lhsT_tile, rhs_tile, transpose_x=True)

    result_sbuf = nl.copy(result_psum, dtype=result.dtype)

    nl.store(result[i_out_p, i_out_f], value=result_sbuf)

    return result

@nki.jit
def tiled_gemm_nki_kernel(lhsT, rhs):
    """
    Tiled matrix multiplication on Trainium using NKI.

    Computes result = lhsT^T @ rhs, i.e. result[m,n] = sum_k lhsT[k,m] * rhs[k,n].

    Args:
        lhsT: (K, M) left operand, pre-transposed so the contraction axis K
              is already the partition (first) dimension.
        rhs:  (K, N) right operand.

    Returns:
        result: (M, N) output matrix.

    Constraints:
        K % 128 == 0
        M % 128 == 0
        N % 512 == 0
    """
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, f"Contraction dimension mismatch: {K} vs {K_}"

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax                  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax      # 512

    assert M % TILE_M == 0, f"M ({M}) must be a multiple of {TILE_M}"
    assert K % TILE_K == 0, f"K ({K}) must be a multiple of {TILE_K}"
    assert N % TILE_N == 0, f"N ({N}) must be a multiple of {TILE_N}"

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            res_psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
                lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

                nisa.dma_copy(dst=lhsT_tile,
                              src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                                       m * TILE_M:(m + 1) * TILE_M])
                nisa.dma_copy(dst=rhs_tile,
                              src=rhs[k * TILE_K:(k + 1) * TILE_K,
                                      n * TILE_N:(n + 1) * TILE_N])

                res_psum += nisa.nc_matmul(lhsT_tile, rhs_tile)

            res_sbuf = nisa.tensor_copy(res_psum, dtype=result.dtype)

            nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                                     n * TILE_N:(n + 1) * TILE_N],
                          src=res_sbuf)

    return result
