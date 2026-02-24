import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):
    
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Tiling Dimensions: Maps MATMUL down to K, M, N 
    IC_TILE = min(in_channels, nl.tile_size.pmax) # 128 -> K = in_channels (contraction dimension)
    OC_TILE = min(out_channels, nl.tile_size.gemm_stationary_fmax) # 128 -> M = out channels
    WPOOL_TILE = min(out_pool_width, nl.tile_size.gemm_moving_fmax) # 512 -> N = image width after pooling
    W_TILE = pool_size * WPOOL_TILE

    # Define number of tiles of each type for loop legibility
    OC_numTiles = out_channels // OC_TILE
    WPOOL_numTiles = out_pool_width // WPOOL_TILE
    IC_numTiles = in_channels // IC_TILE

    # Preload, pretranspose, and pretile weights & biases into sbuf 
    # 128 x 128 x IC_numTile x OC_num_tiles x fh x fw
    w_sbuf_PT = nl.ndarray((OC_TILE, OC_numTiles, in_channels, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    w_sbuf = nl.ndarray((IC_TILE, OC_TILE, IC_numTiles, OC_numTiles, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    bias_sbuf = nl.ndarray((OC_TILE, OC_numTiles), dtype=np.float32, buffer=nl.sbuf)

    for m in nl.affine_range(OC_numTiles):
        # move one (OC_TILE, in_channels, filter_height, filter_width) slice of W into w_sbuf to be transposed
        nisa.dma_copy(dst=w_sbuf_PT[:, m, :, :, :], src = W[m * OC_TILE : (m+1) * OC_TILE, :, :, :])

        # move bias into sbuf
        nisa.dma_copy(dst=bias_sbuf[:, m], src=bias[m * OC_TILE : (m+1) * OC_TILE])

        for k in nl.affine_range(IC_numTiles):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    # iterate over k,i,j since we need a  (128, 128) tile to call nica.nc_transpose
                    transposed = nisa.nc_transpose(w_sbuf_PT[:, m, k * IC_TILE : (k+1) * IC_TILE, i, j])
                    w_sbuf[:, :, k, m, i, j] = nisa.tensor_copy(transposed, dtype=W.dtype)


    for b in nl.affine_range(batch_size):
        # iterate over rows of shape (out_channels, out_width), pooled down to (out_channels, out_pool_width)
        for r in nl.affine_range(out_pool_height):
            # starting row of X necessary for output row r
            input_field_base = r * pool_size

            # load in field of X needed for this row
            # SHAPE = (IC_TILE, IC_numTiles, filter_height + 1, input_width)
            x_sbuf = nl.ndarray((IC_TILE, IC_numTiles, filter_height + (pool_size - 1), input_width), dtype=X.dtype, buffer=nl.sbuf)
            for k in nl.affine_range(IC_numTiles):
                nisa.dma_copy(dst=x_sbuf[:, k, :, :], src=X[b, k * IC_TILE : (k+1) * IC_TILE, input_field_base : input_field_base + filter_height + (pool_size - 1), :])

            # load in output buffer to reduce DMA writes to X_out
            x_out_sbuf = nl.ndarray((OC_TILE, OC_numTiles, WPOOL_TILE * WPOOL_numTiles), dtype=X_out.dtype)
            for m in nl.affine_range(OC_numTiles):
                for n in nl.affine_range(WPOOL_numTiles):
                    # range for N = out_pool_width tile dim
                    col_start = n * WPOOL_TILE
                    col_end = col_start + WPOOL_TILE

                    # Buffer to hold matmul'd data before pooling
                    pre_pool = nl.ndarray((OC_TILE, WPOOL_TILE, pool_size, pool_size), dtype=X.dtype, buffer=nl.sbuf)

                    # Compute two rows if pooling
                    for p in nl.affine_range(pool_size):
                        # initiate PSUM tile to accummulate into
                        res_psum = nl.zeros((OC_TILE, W_TILE), dtype=nl.float32, buffer=nl.psum)
                        for i in nl.affine_range(filter_height):
                            for j in nl.affine_range(filter_width):
                                conv_col_base = pool_size * col_start + j
                                for k in nl.affine_range(IC_numTiles):
                                    # iterate over i, j, IC_TILES (shift operation) to perform matmul and write result to psum
                                    res_psum += nisa.nc_matmul(w_sbuf[:, :, k, m, i, j], x_sbuf[:, k, i + p, conv_col_base : conv_col_base + W_TILE])

                        # (OC_TILE, W_TILE) psum result from one tile -> bring back into SBUF, reshape for pooling, and write back to temporary buffer
                        conv_row_sbuf = nl.copy(res_psum, dtype=X.dtype)
                        tile_reshaped = conv_row_sbuf.reshape((OC_TILE, WPOOL_TILE, pool_size))
                        pre_pool[:, :, :, p] = tile_reshaped

                    # pool entire tile after computing two rows: (OC_TILE, W_TILE, pool_size, pool_size) --> (OC_TILE, WPOOL_TILE)
                    pooled_tile = nisa.tensor_reduce(nl.max, pre_pool, (2, 3))

                    # add bias
                    res_sbuf = nisa.tensor_scalar(pooled_tile, np.add, bias_sbuf[:, m])
                    
                    # Write back out to x_out_sbuf
                    x_out_sbuf[:, m, n * WPOOL_TILE : (n+1) * WPOOL_TILE] = res_sbuf

                # write portion to X_out
                nisa.dma_copy(dst=X_out[b, m * OC_TILE : (m+1) * OC_TILE, r, n * WPOOL_TILE : (n+1) * WPOOL_TILE], src=x_out_sbuf[:, m, n * WPOOL_TILE : (n+1) * WPOOL_TILE])
                
    return X_out

    


    



    
