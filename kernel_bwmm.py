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
    hidden_states,         # Shape: [T, H]             (e.g., [num_tokens, 4096])
    gate_proj,             # Shape: [E, H, I]          (e.g., [8, 4096, 14336])*
    up_proj,               # Shape: [E, H, I]          (e.g., [8, 4096, 14336])*
    down_proj,             # Shape: [E, I, H]          (e.g., [8, 14336, 4096])*
    routing_weights,       # Shape: [T, K]             (e.g., [num_tokens, 2])
    block_token_indices,   # Shape: [M, B]             (e.g., [max_blocks, 128])
    block_expert_ids,      # Shape: [M]                (e.g., [max_blocks])
    output                 # Shape: [T, H]             (e.g., [num_tokens, 4096])
):
    # NKI Kernel Implementation goes here
    return