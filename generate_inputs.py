"""
Generate input tensors for MoE NxD experiments.

Creates two sets of input tensors at sequence lengths 4096, 8192, and 16384:
  1. Uniform distribution - standard random inputs (natural expert spread)
  2. Skewed distribution - inputs optimized to route >80% of tokens to 2 experts

All tensors are [1, seq_len, 4096] in bfloat16, saved to weights/input_weights/.

Usage:
    python generate_inputs.py
"""

import os
import torch
import torch.nn.functional as F

# Mistral 8x7B dimensions
HIDDEN_SIZE = 4096
NUM_EXPERTS = 8
TOP_K = 2

MODEL_WEIGHTS_PATH = "weights/uncompiled_model_weights/mistral_moe.pt"
OUTPUT_DIR = "weights/input_weights"
SEQ_LENGTHS = [4096, 8192, 16384]

# Skewed generation hyperparameters
SKEW_TARGET_EXPERTS = [0, 1]  # Route tokens to expert 0 and expert 1
SKEW_LR = 0.05
SKEW_STEPS = 200
SKEW_TARGET_FRAC = 0.90  # Target: 90% of top-k slots go to the 2 target experts


def load_router_weights(model_path: str) -> torch.Tensor:
    """Load the router weight matrix from saved Mistral MoE weights.
    
    Returns:
        Router weight tensor of shape [num_experts, hidden_size]
    """
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    # The router weight key in our MixtralSparseMoeBlock
    router_weight = state_dict["router.router.weight"]  # [num_experts, hidden_size]
    print(f"  Loaded router weights: {router_weight.shape}, dtype={router_weight.dtype}")
    return router_weight.float()


def compute_expert_distribution(hidden_states: torch.Tensor, router_weight: torch.Tensor) -> dict:
    """Run the router on hidden_states and return expert assignment statistics.
    
    Args:
        hidden_states: [batch, seq_len, hidden_size] float32
        router_weight: [num_experts, hidden_size] float32
    
    Returns:
        Dict with expert_counts, top2_fraction, selected_experts
    """
    flat = hidden_states.view(-1, HIDDEN_SIZE)  # [num_tokens, hidden_size]
    logits = F.linear(flat, router_weight)  # [num_tokens, num_experts]
    top_k_logits, selected_experts = torch.topk(logits, TOP_K, dim=-1)  # [num_tokens, top_k]
    
    # Count how many top-k slots each expert gets
    expert_counts = torch.zeros(NUM_EXPERTS, dtype=torch.long)
    for e in range(NUM_EXPERTS):
        expert_counts[e] = (selected_experts == e).sum().item()
    
    total_slots = selected_experts.numel()
    return {
        "expert_counts": expert_counts,
        "total_slots": total_slots,
        "selected_experts": selected_experts,
    }


def generate_uniform_inputs(seq_len: int) -> torch.Tensor:
    """Generate random inputs with natural (roughly uniform) expert distribution.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Tensor of shape [1, seq_len, HIDDEN_SIZE] in bfloat16
    """
    # Standard normal inputs — the router will naturally spread across experts
    x = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.float32)
    return x.to(torch.bfloat16)


def generate_skewed_inputs(
    seq_len: int,
    router_weight: torch.Tensor,
    target_experts: list[int] = SKEW_TARGET_EXPERTS,
    lr: float = SKEW_LR,
    num_steps: int = SKEW_STEPS,
    target_frac: float = SKEW_TARGET_FRAC,
) -> torch.Tensor:
    """Generate inputs that are skewed to route most tokens to target_experts.
    
    Uses gradient descent to optimize hidden states so the router assigns
    the vast majority of top-k slots to the target experts.
    
    Args:
        seq_len: Sequence length
        router_weight: [num_experts, hidden_size] frozen router weights
        target_experts: List of expert indices to skew toward
        lr: Learning rate for optimization
        num_steps: Number of optimization steps
        target_frac: Stop early if this fraction of slots route to targets
    
    Returns:
        Tensor of shape [1, seq_len, HIDDEN_SIZE] in bfloat16
    """
    # Start from random initialization
    x = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)
    
    target_mask = torch.zeros(NUM_EXPERTS, dtype=torch.float32)
    for e in target_experts:
        target_mask[e] = 1.0
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        flat = x.view(-1, HIDDEN_SIZE)  # [num_tokens, hidden_size]
        logits = F.linear(flat, router_weight)  # [num_tokens, num_experts]
        
        # Loss: maximize logits for target experts, minimize for others
        # Use softmax to get routing probabilities, then maximize target prob
        probs = F.softmax(logits, dim=-1)  # [num_tokens, num_experts]
        target_prob = (probs * target_mask.unsqueeze(0)).sum(dim=-1)  # [num_tokens]
        
        # Negative because we want to maximize
        loss = -target_prob.mean()
        loss.backward()
        optimizer.step()
        
        # Check progress periodically
        if (step + 1) % 50 == 0 or step == 0:
            with torch.no_grad():
                stats = compute_expert_distribution(x.detach(), router_weight)
                target_count = sum(stats["expert_counts"][e].item() for e in target_experts)
                frac = target_count / stats["total_slots"]
                print(f"    Step {step+1}/{num_steps}: target expert fraction = {frac:.3f}, loss = {loss.item():.4f}")
                
                if frac >= target_frac:
                    print(f"    Reached target fraction {target_frac}, stopping early.")
                    break
    
    return x.detach().to(torch.bfloat16)


def verify_distribution(tensor: torch.Tensor, router_weight: torch.Tensor, label: str):
    """Print expert distribution stats for a given input tensor."""
    x = tensor.float()
    stats = compute_expert_distribution(x, router_weight)
    counts = stats["expert_counts"]
    total = stats["total_slots"]
    
    print(f"  [{label}] Expert distribution (top-k slot counts):")
    for e in range(NUM_EXPERTS):
        pct = counts[e].item() / total * 100
        bar = "█" * int(pct / 2)
        print(f"    Expert {e}: {counts[e].item():6d} ({pct:5.1f}%) {bar}")
    print()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load router weights for skewed generation and verification
    print("Loading router weights...")
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"WARNING: Model weights not found at {MODEL_WEIGHTS_PATH}")
        print("  Generating skewed inputs using a fresh random router instead.")
        router_weight = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, dtype=torch.float32) * 0.02
    else:
        router_weight = load_router_weights(MODEL_WEIGHTS_PATH)
    
    print()
    
    for seq_len in SEQ_LENGTHS:
        print(f"{'='*60}")
        print(f"Sequence length: {seq_len}")
        print(f"{'='*60}")
        
        # --- Uniform inputs ---
        print(f"\n  Generating uniform input (seq_len={seq_len})...")
        uniform_tensor = generate_uniform_inputs(seq_len)
        uniform_path = os.path.join(OUTPUT_DIR, f"uniform_b1_s{seq_len}.pt")
        torch.save(uniform_tensor, uniform_path)
        print(f"  Saved: {uniform_path} | shape={uniform_tensor.shape} | dtype={uniform_tensor.dtype}")
        verify_distribution(uniform_tensor, router_weight, f"uniform s={seq_len}")
        
        # --- Skewed inputs ---
        print(f"  Generating skewed input (seq_len={seq_len}, targets={SKEW_TARGET_EXPERTS})...")
        skewed_tensor = generate_skewed_inputs(seq_len, router_weight)
        skewed_path = os.path.join(OUTPUT_DIR, f"skewed_b1_s{seq_len}.pt")
        torch.save(skewed_tensor, skewed_path)
        print(f"  Saved: {skewed_path} | shape={skewed_tensor.shape} | dtype={skewed_tensor.dtype}")
        verify_distribution(skewed_tensor, router_weight, f"skewed s={seq_len}")
    
    print(f"\n{'='*60}")
    print("All input tensors generated successfully!")
    print(f"{'='*60}")
    
    # Summary
    print("\nGenerated files:")
    for seq_len in SEQ_LENGTHS:
        for dist in ["uniform", "skewed"]:
            path = os.path.join(OUTPUT_DIR, f"{dist}_b1_s{seq_len}.pt")
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
