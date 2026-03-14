#!/usr/bin/env python3
"""
Token-Spread MoE Layer — Benchmark on Trainium2

Algorithm (following kernel_v0.py / NKIMoELayer in harness.py):
  1. Router produces top-k expert indices and softmax weights per token
  2. Tokens are replicated (one copy per top-k slot), then scattered into
     per-expert bins using a one-hot cumulative-sum ranking
  3. Each bin is padded to a multiple of TILE_SIZE (128) for alignment
  4. Each expert processes ONLY its assigned tokens (smaller GEMM)
  5. Outputs are gathered back and recombined with routing weights

Compared to dense_gemm_pt.py, this avoids redundant computation on
tokens not routed to a given expert, at the cost of scatter/gather
bookkeeping and padding overhead.

Usage:
    python token_spread_pt.py --batch-size 1 --seq-len 128
    python token_spread_pt.py --batch-size 4 --seq-len 512 --random-weights
    python token_spread_pt.py --batch-size 1 --seq-len 128 --simulate
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_PATH = "weights/mistral_moe.pt"
TILE_SIZE = 128


def _max_tok_per_expert(num_tokens, top_k, num_experts):
    """Padded max tokens per expert bin (expected count, tile-aligned, min 1 tile)."""
    expected = (num_tokens * top_k) / num_experts
    return max(math.ceil(expected / TILE_SIZE) * TILE_SIZE, TILE_SIZE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TokenSpreadMoE(nn.Module):
    """MoE layer using token-spread (sort → pad → per-expert GEMM → unsort).

    Expert weights are stored as stacked 2D buffers
    ``(num_experts * dim_in, dim_out)`` so each expert's slice can be
    extracted with simple arithmetic indexing, keeping shapes static for XLA.
    """

    def __init__(self, hidden_size=4096, intermediate_size=14336,
                 num_experts=8, top_k=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # Stacked expert weights — same layout as NKIMoELayer in harness.py
        self.register_buffer(
            "w1_experts",
            torch.empty(num_experts * hidden_size, intermediate_size))
        self.register_buffer(
            "w3_experts",
            torch.empty(num_experts * hidden_size, intermediate_size))
        self.register_buffer(
            "w2_experts",
            torch.empty(num_experts * intermediate_size, hidden_size))

    def forward(self, hidden_states):
        B, S, D = hidden_states.shape
        N = B * S
        x = hidden_states.view(N, D)

        # === Step 1: Routing — top-k selection then softmax ===
        logits = self.router(x)                              # (N, E)
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        rw = F.softmax(topk_vals, dim=-1, dtype=torch.float).to(x.dtype)

        # === Step 2: Flatten for top-k processing ===
        # Each token is replicated top_k times (once per selected expert)
        flat_exp = topk_idx.reshape(-1).long()               # (N*k,)
        flat_wt  = rw.reshape(-1)                            # (N*k,)
        repeated = (x.unsqueeze(1)
                      .expand(-1, self.top_k, -1)
                      .reshape(-1, D))                       # (N*k, D)

        # === Step 3: Within-expert rank via one-hot cumulative sum ===
        # Assigns each token a position inside its expert's bin while
        # keeping all shapes static (no dynamic boolean indexing).
        max_tok = _max_tok_per_expert(N, self.top_k, self.num_experts)
        T_pad = self.num_experts * max_tok

        oh   = F.one_hot(flat_exp, self.num_experts).float()
        rank = (torch.cumsum(oh, dim=0) * oh).sum(-1).long() - 1
        rank = rank.clamp(max=max_tok - 1)
        dest = flat_exp * max_tok + rank                     # (N*k,)

        # === Step 4: Scatter tokens and weights into padded expert bins ===
        pad_tok = torch.zeros(T_pad, D, dtype=x.dtype, device=x.device)
        pad_tok.scatter_(0, dest.unsqueeze(1).expand(-1, D), repeated)

        pad_wt = torch.zeros(T_pad, dtype=x.dtype, device=x.device)
        pad_wt.scatter_(0, dest, flat_wt)

        # === Step 5: Per-expert FFN (only assigned tokens per expert) ===
        # Each expert processes max_tok tokens (its bin), NOT all N tokens.
        # Padding tokens are zero and produce zero output.
        expert_out = torch.zeros(T_pad, D, dtype=torch.float32,
                                 device=x.device)

        for e in range(self.num_experts):
            start = e * max_tok
            end   = (e + 1) * max_tok
            tok   = pad_tok[start:end].float()

            w1_e = self.w1_experts[e * D:(e + 1) * D].float()
            w3_e = self.w3_experts[e * D:(e + 1) * D].float()
            w2_e = self.w2_experts[
                e * self.intermediate_size:(e + 1) * self.intermediate_size
            ].float()

            expert_out[start:end] = (F.silu(tok @ w1_e) * (tok @ w3_e)) @ w2_e

        expert_out = expert_out.to(x.dtype)

        # === Step 6: Recombine — apply weights, gather, sum over top-k ===
        weighted = expert_out.float() * pad_wt.float().unsqueeze(1)
        gathered = torch.gather(weighted, 0,
                                dest.unsqueeze(1).expand(-1, D))
        final = gathered.reshape(N, self.top_k, D).sum(dim=1)

        output = final.to(x.dtype).view(B, S, D)
        return output, topk_idx.view(B, S, self.top_k), rw.view(B, S, self.top_k)

    # --- Factory methods -------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_path=MODEL_PATH):
        """Load weights from a saved MixtralSparseMoeBlock state dict.

        Transposes per-expert nn.Linear weights from (out, in) to (in, out)
        and stacks them into the contiguous 2D layout this class expects.
        """
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

        w1, w2, w3 = [], [], []
        for expert in block.experts:
            w1.append(expert.w1.weight.data.T)   # (out,in) → (in,out)
            w2.append(expert.w2.weight.data.T)
            w3.append(expert.w3.weight.data.T)

        model.w1_experts.copy_(
            torch.stack(w1).reshape(-1, mx_cfg.intermediate_size))
        model.w3_experts.copy_(
            torch.stack(w3).reshape(-1, mx_cfg.intermediate_size))
        model.w2_experts.copy_(
            torch.stack(w2).reshape(-1, mx_cfg.hidden_size))

        return model.to(dtype=torch.bfloat16)

    @classmethod
    def random_init(cls, hidden_size=512, intermediate_size=1024,
                    num_experts=8, top_k=2):
        """Create with Kaiming-uniform random weights."""
        model = cls(hidden_size, intermediate_size, num_experts, top_k)
        nn.init.kaiming_uniform_(model.router.weight)
        nn.init.kaiming_uniform_(model.w1_experts)
        nn.init.kaiming_uniform_(model.w3_experts)
        nn.init.kaiming_uniform_(model.w2_experts)
        return model.to(dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# CLI + benchmark harness
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Token-Spread MoE — benchmark on Trainium2")
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
    print("Token-Spread MoE — Trainium2 Benchmark")
    print("=" * 60)

    # --- Build model ---
    use_pretrained = not args.random_weights and os.path.exists(MODEL_PATH)
    if use_pretrained:
        print(f"Loading pretrained weights from {MODEL_PATH}")
        model = TokenSpreadMoE.from_pretrained(MODEL_PATH)
    else:
        print("Using random weights")
        model = TokenSpreadMoE.random_init()

    model.eval()
    max_tok = _max_tok_per_expert(B * S, model.top_k, model.num_experts)
    T_pad = model.num_experts * max_tok

    print(f"  hidden={model.hidden_size}  intermediate={model.intermediate_size}  "
          f"experts={model.num_experts}  top_k={model.top_k}")
    print(f"  batch_size={B}  seq_len={S}  tokens={B * S}")
    print(f"  max_tok_per_expert={max_tok}  T_padded={T_pad}")

    inputs = torch.randn(B, S, model.hidden_size, dtype=torch.bfloat16)

    # --- Forward pass ---
    if args.simulate:
        print("\nRunning Token-Spread MoE on CPU ...")
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
        print("Running Token-Spread MoE on Trainium ...")
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
