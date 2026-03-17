import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import nki
import nki.language as nl
import nki.isa as nisa

import torch
from torch_xla.core import xla_model as xm

MODEL_PATH = "weights/compiled_model_weights/mistral_moe.pt"
INPUT_TENSOR_PATH = "weights/input_weights/prefill_b1_s4096.pt"
OUTPUT_TENSOR_PATH = "weights/output_weights/prefill_b1_s4096.pt"
BLOCK_SIZE = 128

# Variable Definitions for Documentation:
# T: num_tokens (batch_size * sequence_length)
# E: num_experts (e.g., 8)
# K: top_k (e.g., 2)
# H: hidden_dim (4096)
# I: intermediate_dim (14336)
# B: block_size (128)
# M: max_blocks (The static integer calculated via min(bound_1, bound_2))

@nki.jit
def kernel(
    hidden_states,          # [4096, num_tokens] (Transposed Tokens)
    w1,                     # [num_experts, 4096, 14336]
    w3,                     # [num_experts, 4096, 14336]
    w2,                     # [num_experts, 14336, 4096]
    routing_weights,        # T x K
    flattened_blocks,       # [MAX_BLOCKS, 128]  ---> the blocks themselves
    expert_block_offsets,   # [num_experts] --------> where the blocks for each expert start in flattened blocks
    expert_block_counts,    # [num_experts] --------> # how many blocks per expert
    output,                 # [num_tokens, 4096]
    max_blocks_per_expert
):
    # get constant dim values
    # feature_dim, num_tokens = hidden_states.shape
    # num_experts, _, intermediate_dim = w1.shape
    # max_blocks, block_size = flattened_blocks.shape

    # --- STATIC HARDWARE LIMITS ---
    MAX_BLOCKS = 12 
    TILE_DIM = 128
    NUM_H_TILES = 4096 // TILE_DIM  # 32 tiles to cover the hidden dimension
    
   # 1. The Outer Expert Loop (Static: 0 to 7)
    for e in nl.affine_range(8):
        
        # --- PHASE A: METADATA & TOKEN HOISTING ---
        
        # Lock the dynamic block bounds for THIS specific expert into hardware registers
        start_block = nisa.register_alloc(expert_block_offsets[e])
        num_blocks = nisa.register_alloc(expert_block_counts[e])
        
        # Allocate our massive, static SBUF structure to hold the hoisted tokens.
        # Because of hardware limits, we create a 2D Python list of [128, 128] nl.ndarrays.
        # Access pattern: expert_tokens_sbuf[block_index][hidden_tile_index]
        expert_tokens_sbuf = []
        for b_idx in range(MAX_BLOCKS):
            block_list = []
            for h_idx in range(NUM_H_TILES):
                block_list.append(nl.ndarray(shape=(TILE_DIM, TILE_DIM), 
                                             dtype=hidden_states.dtype, 
                                             buffer=nl.sbuf))
            expert_tokens_sbuf.append(block_list)
        
        # DYNAMIC LOOP: Fetch only the actual tokens assigned to this expert from HBM
        for b in nl.dynamic_range(num_blocks):
            global_block_idx = start_block + b
            
            # 1. Fetch the 128 integer token indices for this specific block into SBUF
            indices_sbuf = nl.ndarray(shape=(TILE_DIM,), dtype=nl.int32, buffer=nl.sbuf)
            nl.dma_copy(dst=indices_sbuf, src=flattened_blocks[global_block_idx, :])
            
            # 2. Use the DGE to gather the 4096 hidden features for those 128 tokens
            for h in nl.affine_range(NUM_H_TILES):
                # Calculate the exact row slice for this hidden tile
                h_start = h * TILE_DIM
                h_end = (h + 1) * TILE_DIM
                
                # INDIRECT DMA COPY: By passing `indices_sbuf` into the column index, 
                # the hardware's DGE automatically jumps around HBM to fetch the 
                # scattered tokens and packs them into our dense SBUF tile!
                nl.dma_copy(dst=expert_tokens_sbuf[b][h], src=hidden_states[h_start:h_end, indices_sbuf])

        # --- PHASE B: GATE/UP PROJECTIONS ---
        pass
        
        # --- PHASE C: IN-SBUF ACTIVATION & DOWN PROJECTION ---
        pass
        
    return