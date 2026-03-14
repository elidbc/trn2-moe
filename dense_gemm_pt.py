#!/usr/bin/env python3
"""
Dense GEMM MoE Layer — Benchmark on Trainium2

Algorithm (following mixtral_references/mistral_moe.py):
  1. Router produces top-k expert indices and softmax weights per token
  2. Every expert processes ALL tokens (dense GEMM — no dynamic shapes)
  3. Outputs are zeroed via a routing mask for non-selected experts
  4. Weighted results are accumulated across experts

This is the simplest XLA-compatible approach: no scatter/gather, no
dynamic shapes, but it performs redundant computation on tokens that
were not routed to a given expert.

Usage:
    python dense_gemm_pt.py --batch-size 1 --seq-len 128
    python dense_gemm_pt.py --batch-size 4 --seq-len 512 --random-weights
    python dense_gemm_pt.py --batch-size 1 --seq-len 128 --simulate
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_PATH = "weights/mistral_moe.pt"


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    """Single expert MLP: SwiGLU(x @ W1) * (x @ W3) then @ W2."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DenseGemmMoE(nn.Module):
    """MoE layer using dense GEMM + masking.

    Every expert sees every token. Non-routed outputs are zeroed by
    multiplying with a mask derived from the router's top-k selection.
    This avoids all dynamic indexing, making it fully XLA-safe.
    """

    def __init__(self, hidden_size=4096, intermediate_size=14336,
                 num_experts=8, top_k=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, hidden_states):
        B, S, D = hidden_states.shape
        x = hidden_states.view(-1, D)                       # (N, D)

        # --- Routing: top-k selection then softmax ---
        logits = self.router(x)                              # (N, num_experts)
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        routing_weights = F.softmax(topk_vals, dim=-1, dtype=torch.float).to(x.dtype)

        # One-hot mask: (num_experts, N, top_k) — which top-k slot selected
        # each expert for each token
        expert_mask = F.one_hot(
            topk_idx.long(), self.num_experts
        ).permute(2, 0, 1)

        # --- Dense expert computation with masking ---
        final = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            # Dense GEMM: run this expert on ALL tokens
            expert_out = self.experts[expert_idx](x)

            # Build per-token scalar weight for this expert:
            #   sum of routing_weights where this expert was selected (0 otherwise)
            weight_mask = (routing_weights * expert_mask[expert_idx]).sum(dim=-1)

            # Zero out non-routed tokens, scale routed ones
            final += expert_out * weight_mask.unsqueeze(-1)

        output = final.view(B, S, D)
        return output, topk_idx.view(B, S, self.top_k), routing_weights.view(B, S, self.top_k)

    # --- Factory methods -------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_path=MODEL_PATH):
        """Load weights from a saved MixtralSparseMoeBlock state dict."""
        from mixtral_references.mistral_moe import (
            MixtralConfig, MixtralSparseMoeBlock,
        )

        mx_cfg = MixtralConfig()
        state_dict = torch.load(model_path, map_location="cpu",
                                weights_only=True)

        block = MixtralSparseMoeBlock(mx_cfg)
        block.load_state_dict(state_dict)
        block = block.to(dtype=torch.bfloat16)

        model = cls(
            mx_cfg.hidden_size, mx_cfg.intermediate_size,
            mx_cfg.num_local_experts, mx_cfg.num_experts_per_tok,
        )

        model.router.weight.data.copy_(block.router.router.weight.data)
        for i, expert in enumerate(block.experts):
            model.experts[i].w1.weight.data.copy_(expert.w1.weight.data)
            model.experts[i].w2.weight.data.copy_(expert.w2.weight.data)
            model.experts[i].w3.weight.data.copy_(expert.w3.weight.data)

        return model.to(dtype=torch.bfloat16)

    @classmethod
    def random_init(cls, hidden_size=512, intermediate_size=1024,
                    num_experts=8, top_k=2):
        """Create with default-initialized random weights."""
        return cls(hidden_size, intermediate_size,
                   num_experts, top_k).to(dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# CLI + benchmark harness
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dense GEMM MoE — benchmark on Trainium2")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size [default: 1]")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length [default: 128]")
    parser.add_argument("--random-weights", action="store_true",
                        help="Use random weights (default: load "
                             "weights/mistral_moe.pt if available)")
    parser.add_argument("--simulate", action="store_true",
                        help="Run on CPU only (no Trainium required)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed [default: 42]")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    B, S = args.batch_size, args.seq_len

    print("=" * 60)
    print("Dense GEMM MoE — Trainium2 Benchmark")
    print("=" * 60)

    # --- Build model ---
    use_pretrained = not args.random_weights and os.path.exists(MODEL_PATH)
    if use_pretrained:
        print(f"Loading pretrained weights from {MODEL_PATH}")
        model = DenseGemmMoE.from_pretrained(MODEL_PATH)
    else:
        print("Using random weights")
        model = DenseGemmMoE.random_init()

    model.eval()
    print(f"  hidden={model.hidden_size}  intermediate={model.intermediate_size}  "
          f"experts={model.num_experts}  top_k={model.top_k}")
    print(f"  batch_size={B}  seq_len={S}  tokens={B * S}")

    inputs = torch.randn(B, S, model.hidden_size, dtype=torch.bfloat16)

    # --- Forward pass ---
    if args.simulate:
        print("\nRunning Dense GEMM MoE on CPU ...")
        with torch.no_grad():
            t0 = time.perf_counter()
            output, topk_idx, rw = model(inputs)
            t1 = time.perf_counter()
        print(f"  Elapsed: {(t1 - t0) * 1000:.2f} ms")
    else:
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = torch_xla.device()
        model = model.to(device)
        inputs_dev = inputs.to(device)

        # Warmup (triggers XLA compilation)
        print("\nWarming up (XLA compile) ...")
        with torch.no_grad():
            _ = model(inputs_dev)
        torch_xla.sync()

        # Timed run
        print("Running Dense GEMM MoE on Trainium ...")
        with torch.no_grad():
            t0 = time.perf_counter()
            output, topk_idx, rw = model(inputs_dev)
            torch_xla.sync()
            t1 = time.perf_counter()

        output = output.cpu()
        print(f"  Elapsed: {(t1 - t0) * 1000:.2f} ms")

    print(f"  Output shape: {list(output.shape)}")
    print(f"  Output range: [{output.float().min().item():.4f}, "
          f"{output.float().max().item():.4f}]")
    print("\nDone.")


if __name__ == "__main__":
    main()
