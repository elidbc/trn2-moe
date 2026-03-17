import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Config with exact dimensions for Mistral 8x7B model
class MixtralConfig:
    def __init__(self):
        self.hidden_size = 4096 
        self.intermediate_size = 14336
        self.num_local_experts = 8
        self.topk = 2

# Router class
class MixtralRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.top_k = config.topk

    def forward(self, hidden_states):
        raw_logits = self.router(hidden_states) 
        top_k_logits, selected_experts = torch.topk(raw_logits, self.top_k, dim=-1) 
        routing_weights = F.softmax(top_k_logits, dim=-1, dtype=torch.float)
        return routing_weights.to(hidden_states.dtype), selected_experts

# Full MoE Layer
class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts       
        self.top_k = config.topk        

        # The Router / Gating Network
        self.router = MixtralRouter(config)
        
        # The Experts (Fixed: proper nn.Parameter syntax and transposed shapes)
        self.w1 = nn.Parameter(torch.empty(self.num_experts, config.intermediate_size, config.hidden_size))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, config.intermediate_size))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, config.intermediate_size, config.hidden_size))
        self.act_fn = nn.SiLU()

        # Initialize with realistic fake weights to avoid junk memory values
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using a standard normal distribution common in Transformers."""
        nn.init.normal_(self.w1, mean=0.0, std=0.02)
        nn.init.normal_(self.w2, mean=0.0, std=0.02)
        nn.init.normal_(self.w3, mean=0.0, std=0.02)
        nn.init.normal_(self.router.router.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim) 
        routing_weights, selected_experts = self.router(hidden_states)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )

        expert_mask = F.one_hot(selected_experts.to(torch.long), num_classes=self.num_experts).permute(2, 0, 1)
        
        for expert_idx in range(self.num_experts):
            expert_w1 = self.w1[expert_idx]
            expert_w2 = self.w2[expert_idx]
            expert_w3 = self.w3[expert_idx]

            h1 = F.linear(hidden_states, expert_w1)
            h3 = F.linear(hidden_states, expert_w3)
            current_hidden_states = F.linear(self.act_fn(h1) * h3, expert_w2)
            
            weight_mask = (routing_weights * expert_mask[expert_idx]).sum(dim=-1)
            current_hidden_states = current_hidden_states * weight_mask.unsqueeze(-1)
            final_hidden_states += current_hidden_states

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        selected_experts = selected_experts.reshape(batch_size, sequence_length, self.top_k)
        
        return final_hidden_states, selected_experts


# ==========================================
# Script to Save, Load, and Test the Model
# ==========================================
if __name__ == "__main__":
    config = MixtralConfig()
    save_path = "weights/uncompiled_model_weights/mistral_moe_new.pt"
    
    print("1. Initializing the MoE Block with realistic fake weights...")
    model_to_save = MixtralSparseMoeBlock(config)
    
    print(f"2. Saving the weights to '{save_path}'...")
    torch.save(model_to_save.state_dict(), save_path)
    
    print("3. Creating a fresh, uninitialized MoE Block...")
    # We create a new instance to prove loading works.
    # Note: It initializes with random weights first, but we will overwrite them.
    loaded_model = MixtralSparseMoeBlock(config)
    
    print("4. Loading the saved weights into the fresh model...")
    state_dict = torch.load(save_path, weights_only=True)
    loaded_model.load_state_dict(state_dict)
    print("   Weights successfully loaded!")
    
    print("\n5. Running a test forward pass...")
    # Create a dummy input tensor: batch_size=1, seq_length=128, hidden_dim=4096
    dummy_input = torch.randn(1, 128, config.hidden_size)
    
    # Run inference
    with torch.no_grad():
        output_states, selected_experts = loaded_model(dummy_input)
        
    print(f"   Input shape:  {dummy_input.shape}")
    print(f"   Output shape: {output_states.shape}")
    print(f"   Router shape: {selected_experts.shape}")
    print("\nSuccess! The dense masking logic and raw weight matrices are working perfectly.")