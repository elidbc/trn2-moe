import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtralBlockSparseTop2MLP(nn.Module):
    """An individual Expert MLP using SwiGLU activation."""
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

class MixtralSparseMoeBlock(nn.Module):
    """The MoE Routing Block (replaces the standard Feed-Forward Network)."""
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts        # For Mixtral 8x7B, this is 8
        self.top_k = config.num_experts_per_tok            # For Mixtral 8x7B, this is 2

        # The Router / Gating Network
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        
        # The Experts (8 separate MLPs)
        self.experts = nn.ModuleList([
            MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # Flatten batch and sequence dimensions for easier routing
        hidden_states = hidden_states.view(-1, hidden_dim) 
        
        # 1. Get router logits and probabilities
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        
        # 2. Select top-k (top-2) experts per token
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # 3. Normalize routing weights so the selected top-k sum to 1
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Initialize the output tensor with zeros
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )

        # 4. Create a one-hot mask to map tokens to their assigned experts
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        # 5. Dispatch tokens to experts and accumulate the results
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
        
        # Return both the processed hidden states and the router logits (often needed for auxiliary loss)
        return final_hidden_states, router_logits


# --- New Additional Code ---

class MixtralConfig:
    """Mock config class to hold the dimensions required by the MoE block."""
    def __init__(self):
        # Using standard Mixtral 8x7B hidden dimensions. 
        # Note: If you OOM on a smaller GPU, cut these down (e.g., 1024 and 4096).
        self.hidden_size = 4096 
        self.intermediate_size = 14336
        self.num_local_experts = 8
        self.num_experts_per_tok = 2

def main():
    config = MixtralConfig()
    weights_path = "mixtral_moe_dummy_weights.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # 1. Instantiate the MoE block
    moe_block = MixtralSparseMoeBlock(config)
    
    # 2. Save/Load Weights Logic
    if os.path.exists(weights_path):
        print(f"Loading existing weights from '{weights_path}'...")
        # Load weights to CPU first to avoid VRAM spikes, then transfer to model
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        moe_block.load_state_dict(state_dict)
    else:
        print(f"Generating new dummy weights and saving to '{weights_path}'...")
        # PyTorch automatically initializes nn.Linear with random weights upon instantiation
        torch.save(moe_block.state_dict(), weights_path)
        
    # 3. Cast to bfloat16 and move to device
    moe_block = moe_block.to(device=device, dtype=torch.bfloat16)
    moe_block.eval() # Set to evaluation mode
    
    # 4. Create Dummy Input
    batch_size = 2
    seq_len = 16
    print(f"Creating dummy input of shape ({batch_size}, {seq_len}, {config.hidden_size}) in bfloat16...")
    
    # Generate random normally distributed values and cast to bf16
    dummy_input = torch.randn(
        batch_size, seq_len, config.hidden_size, 
        dtype=torch.bfloat16, device=device
    )
    
    # 5. Run the block
    print("Running inputs through the MoE block...")
    with torch.no_grad(): # Disable gradient tracking for inference
        output, router_logits = moe_block(dummy_input)
        
    # 6. Output Results
    print("\n--- Execution Successful ---")
    print(f"Output shape: {output.shape} | Expected: ({batch_size}, {seq_len}, {config.hidden_size})")
    print(f"Router logits shape: {router_logits.shape} | Expected: ({batch_size * seq_len}, {config.num_local_experts})")
    
    # Show a tiny slice of the data to verify it's working and in bfloat16
    print(f"\nSample output (first token, first 5 dimensions):")
    print(output[0, 0, :5])

if __name__ == "__main__":
    main()