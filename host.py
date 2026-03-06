import torch
import numpy as np
import math

import nki
import nki.language as nl
import nki.isa as nisa

from mistral_moe import MixtralConfig, MixtralSparseMoeBlock, MixtralRouter, MixtralExpert

MODEL_PATH = "weights/uncompiled_model_weights/mistral_moe.pt"

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