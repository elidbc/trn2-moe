import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py

# Config with exact dimensions for Mistral 8x7B model
class MixtralConfig:
    def __init__(self):
        self.hidden_size = 4096 
        self.intermediate_size = 14336
        self.num_local_experts = 8
        self.num_experts_per_tok = 2


# Router class (maps hidden state inputs to expert logits)
# Note: this is slightly more efficient than the base MixtralRouter class because it takes the top-k before softmax
class MixtralRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states):
        # hidden_states = batch_size * sequence_length, hidden_dim
        raw_logits = self.router(hidden_states) # router_logits = batch_size * sequence_length, num_experts
        top_k_logits, selected_experts = torch.topk(raw_logits, self.top_k, dim=-1) # top_k_logits, selected_experts = batch_size * sequence_length, top_k
        routing_weights = F.softmax(top_k_logits, dim=-1, dtype=torch.float)
        return routing_weights.to(hidden_states.dtype), selected_experts


# Expert class (MLP with SwiGLU activation)
class MixtralExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        # SwiGLU activation: SiLU(xW1) * xW3
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        return self.w2(current_hidden_states)


# Full MoE Layer (combines router and experts)
class MixtralSparseMoeBlockNaive(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts       
        self.top_k = config.num_experts_per_tok         

        # The Router / Gating Network
        self.router = MixtralRouter(config)
        
        # The Experts (8 separate MLPs)
        self.experts = nn.ModuleList([
            MixtralExpert(config) for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Flatten batch and sequence dimensions for easier routing
        hidden_states = hidden_states.view(-1, hidden_dim) 
        
        # Get routing weights and selected experts via the router
        # both will be of shape (batch_size * sequence_length, top_k)
        routing_weights, selected_experts = self.router(hidden_states)

        # Initialize the output tensor with zeros
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )

        # Create a one-hot mask to map tokens to their assigned experts
        expert_mask = F.one_hot(selected_experts.to(torch.long), num_classes=self.num_experts).permute(2, 1, 0)
        
        # Dispatch tokens to experts and accumulate the results
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # Find the indices of tokens routed to the current expert
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Skip execution if no tokens were routed to this expert
            if top_x.shape[0] == 0:
                continue

            # Extract the actual token embeddings assigned to this expert
            current_state = hidden_states[top_x]
            
            # Pass the tokens through the expert MLP
            current_hidden_states = expert_layer(current_state)
            
            # Weight the expert's output by the routing probability
            current_hidden_states = current_hidden_states * routing_weights[top_x, idx].unsqueeze(-1)
            
            # Accumulate the results back into the final output tensor using scatter add
            final_hidden_states.index_add_(0, top_x, current_hidden_states)

        # Reshape back to the original dimensions
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        # Return only final hidden states
        return final_hidden_states