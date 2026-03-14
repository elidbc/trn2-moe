#!/usr/bin/env python3
"""
Blockwise Dropless MoE Layer — Benchmark on Trainium2

Algorithm:
  1. Router produces top-k expert indices and softmax weights per token
  2. Per-expert cumsum ranking assigns each token a slot inside its
     expert's block range — no argsort, no one_hot [M, E] cumsum
  3. Each block belongs to exactly one expert; multiple blocks per expert
  4. Total blocks are statically provisioned:
       n_blocks = ceil_div(M, block_size) + (num_experts - 1)
  5. Per-expert SwiGLU FFN compute over each expert's contiguous blocks
  6. Outputs are gathered back and recombined with routing weights

Compared to token_spread_pt.py, this avoids the giant one_hot(M, E) + cumsum
graph by doing E separate cumsums on [M] masks.  The block structure gives
XLA-friendly static shapes while staying dropless.  Avoids torch.argsort
and torch.bincount, which lower to HLO sort (unsupported on trn2).

Usage:
    python bucketed_pt.py --batch-size 1 --seq-len 128
    python bucketed_pt.py --batch-size 4 --seq-len 512 --random-weights
    python bucketed_pt.py --batch-size 1 --seq-len 128 --simulate
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_PATH = "weights/mistral_moe.pt"


def ceil_div(a, b):
    return (a + b - 1) // b


def provision_n_blocks(M, block_size, num_experts):
    """Statically provisioned number of blocks (upper bound on actual usage)."""
    return ceil_div(M, block_size) + (num_experts - 1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BlockedMoE(nn.Module):
    """MoE layer using blockwise dropless packing.

    Per-expert cumsum ranking assigns each token a position inside its
    expert's block range, then tokens are scattered into a flat blocked
    buffer.  Each expert's blocks are processed as a single matmul,
    and outputs are gathered back.  The total number of blocks is
    statically provisioned so the compilation graph has no dynamic shapes.

    Expert weights are stored as stacked 2D buffers
    ``(num_experts * dim_in, dim_out)`` matching token_spread_pt.py.
    """

    def __init__(self, hidden_size=4096, intermediate_size=14336,
                 num_experts=8, top_k=2, block_size=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.block_size = block_size

        self.router = nn.Linear(hidden_size, num_experts, bias=False)

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
        I = self.intermediate_size
        E = self.num_experts
        bs = self.block_size
        x = hidden_states.view(N, D)

        # === Step 1: Routing — top-k selection then softmax ===
        logits = self.router(x)                                   # (N, E)
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        routing_weights = F.softmax(
            topk_vals, dim=-1, dtype=torch.float).to(x.dtype)

        # === Step 2: Flatten top-k assignments ===
        M = N * self.top_k
        flat_exp = topk_idx.reshape(M).long()
        flat_wt  = routing_weights.reshape(M)
        repeated_tokens = (x.unsqueeze(1)
                           .expand(N, self.top_k, D).reshape(M, D))

        # === Step 3: Per-expert counts ===
        # Per-expert comparison + sum replaces bincount / argsort, both of
        # which lower to HLO sort (unsupported on trn2).
        expert_masks = []
        counts_list = []
        for e in range(E):
            mask_e = (flat_exp == e).long()
            expert_masks.append(mask_e)
            counts_list.append(mask_e.sum())
        counts = torch.stack(counts_list)

        # === Step 4: Block layout ===
        n_blocks = provision_n_blocks(M, bs, E)
        num_blocks_per_expert = (counts + bs - 1) // bs

        # cumsum in float32 — trn2 does not support int64 dot products,
        # and XLA lowers cumsum to a dot with a lower-triangular matrix.
        expert_block_start = torch.cat([
            torch.zeros(1, dtype=torch.long, device=x.device),
            torch.cumsum(
                num_blocks_per_expert[:-1].float(), dim=0).long(),
        ])

        # === Step 5: Slot computation via per-expert cumsum ranking ===
        # Each token's within-expert rank is computed by a cumsum on that
        # expert's [M] mask.  Rank → (local_block, local_pos) → global slot.
        # This replaces argsort + sorted-array arithmetic with E=8
        # cumsums on [M] vectors — no sort, no one_hot [M, E].
        # cumsums run in float32 to avoid int64 dot on trn2.
        slot = torch.zeros(M, dtype=torch.long, device=x.device)
        for e in range(E):
            mask_e = expert_masks[e]
            rank_e = (torch.cumsum(mask_e.float(), dim=0)
                      .long() - 1).clamp(min=0)
            local_block = rank_e // bs
            local_pos   = rank_e % bs
            global_block = expert_block_start[e] + local_block
            slot_e = global_block * bs + local_pos
            slot = slot + slot_e * mask_e

        # === Step 6: Scatter into blocked buffer ===
        total_slots = n_blocks * bs

        blocked_tokens_flat = torch.zeros(
            total_slots, D, dtype=x.dtype, device=x.device)
        blocked_valid_flat = torch.zeros(
            total_slots, dtype=x.dtype, device=x.device)

        blocked_tokens_flat.scatter_(
            0, slot.unsqueeze(1).expand(-1, D), repeated_tokens)
        blocked_valid_flat.scatter_(
            0, slot, torch.ones(M, dtype=x.dtype, device=x.device))

        blocked_tokens = blocked_tokens_flat.view(n_blocks, bs, D)
        blocked_valid   = blocked_valid_flat.view(n_blocks, bs)

        # Block-to-expert mapping
        block_expert = torch.zeros(
            n_blocks, dtype=torch.long, device=x.device)
        ebs_list  = expert_block_start.tolist()
        nbpe_list = num_blocks_per_expert.tolist()
        for e in range(E):
            s, nb = ebs_list[e], nbpe_list[e]
            if nb > 0:
                block_expert[s:s + nb] = e

        # === Step 7: Per-expert MLP (SwiGLU in float32) ===
        # Loop over experts (E=8), each expert processes all its contiguous
        # blocks as a single matmul.  Padding tokens are zero → zero output.
        block_out_flat = torch.zeros(
            total_slots, D, dtype=torch.float32, device=x.device)

        for e in range(E):
            nb = nbpe_list[e]
            if nb == 0:
                continue
            start = ebs_list[e] * bs
            end   = start + nb * bs
            tok  = blocked_tokens_flat[start:end].float()         # (nb*bs, D)
            w1_e = self.w1_experts[e * D:(e + 1) * D].float()    # (D, I)
            w3_e = self.w3_experts[e * D:(e + 1) * D].float()    # (D, I)
            w2_e = self.w2_experts[e * I:(e + 1) * I].float()    # (I, D)
            block_out_flat[start:end] = (
                F.silu(tok @ w1_e) * (tok @ w3_e)) @ w2_e

        block_out_flat = block_out_flat.to(x.dtype)

        # === Step 8: Gather and recombine ===
        # Tokens were never reordered, so no inverse permutation needed —
        # just gather each token's output from its slot.
        routed_out = torch.gather(
            block_out_flat, 0, slot.unsqueeze(1).expand(-1, D))

        weighted_out = routed_out.float() * flat_wt.float().unsqueeze(1)
        final = weighted_out.reshape(N, self.top_k, D).sum(dim=1)
        output = final.to(x.dtype).view(B, S, D)

        return (output,
                topk_idx.view(B, S, self.top_k),
                routing_weights.view(B, S, self.top_k))

    # --- Factory methods -------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_path=MODEL_PATH, block_size=512):
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
            block_size=block_size,
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
                    num_experts=8, top_k=2, block_size=512):
        """Create with Kaiming-uniform random weights."""
        model = cls(hidden_size, intermediate_size, num_experts, top_k,
                    block_size=block_size)
        nn.init.kaiming_uniform_(model.router.weight)
        nn.init.kaiming_uniform_(model.w1_experts)
        nn.init.kaiming_uniform_(model.w3_experts)
        nn.init.kaiming_uniform_(model.w2_experts)
        return model.to(dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Correctness check (BlockedMoE vs DenseGemmMoE on a tiny shape)
# ---------------------------------------------------------------------------

def _correctness_check(block_size=4):
    """Instantiate DenseGemmMoE and BlockedMoE with matching random weights
    on a tiny shape and report max absolute / relative error."""
    from dense_gemm_pt import DenseGemmMoE

    torch.manual_seed(0)
    D, I_size, E, K = 64, 128, 4, 2
    B, S = 1, 16

    dense = DenseGemmMoE(D, I_size, E, K).to(torch.bfloat16)
    dense.eval()

    blocked = BlockedMoE(D, I_size, E, K, block_size=block_size)
    blocked.router.weight.data.copy_(dense.router.weight.data)
    for i in range(E):
        blocked.w1_experts.data[i * D:(i + 1) * D] = (
            dense.experts[i].w1.weight.data.T)
        blocked.w3_experts.data[i * D:(i + 1) * D] = (
            dense.experts[i].w3.weight.data.T)
        blocked.w2_experts.data[i * I_size:(i + 1) * I_size] = (
            dense.experts[i].w2.weight.data.T)
    blocked = blocked.to(torch.bfloat16)
    blocked.eval()

    x = torch.randn(B, S, D, dtype=torch.bfloat16)
    with torch.no_grad():
        out_dense,   idx_d, rw_d = dense(x)
        out_blocked, idx_b, rw_b = blocked(x)

    assert torch.equal(idx_d, idx_b), "Routing indices differ!"
    assert torch.allclose(rw_d, rw_b), "Routing weights differ!"

    abs_err = (out_dense.float() - out_blocked.float()).abs()
    max_abs = abs_err.max().item()
    max_rel = (abs_err / (out_dense.float().abs() + 1e-8)).max().item()

    print(f"Correctness check  (B={B}, S={S}, D={D}, E={E}, "
          f"block_size={block_size}):")
    print(f"  Max absolute error: {max_abs:.6e}")
    print(f"  Max relative error: {max_rel:.6e}")
    ok = max_abs < 5e-2
    print(f"  {'PASSED' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# CLI + benchmark harness
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Blockwise Dropless MoE — benchmark on Trainium2")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size [default: 1]")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length [default: 128]")
    parser.add_argument("--block-size", type=int, default=512,
                        help="Block size for packing [default: 512]")
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
    print("Blockwise Dropless MoE — Trainium2 Benchmark")
    print("=" * 60)

    # --- Build model ---
    use_pretrained = not args.random_weights and os.path.exists(MODEL_PATH)
    if use_pretrained:
        print(f"Loading pretrained weights from {MODEL_PATH}")
        model = BlockedMoE.from_pretrained(MODEL_PATH,
                                           block_size=args.block_size)
    else:
        print("Using random weights")
        model = BlockedMoE.random_init(block_size=args.block_size)

    model.eval()
    N = B * S
    M = N * model.top_k
    n_blocks = provision_n_blocks(M, model.block_size, model.num_experts)
    padded_tokens = n_blocks * model.block_size

    print(f"  hidden={model.hidden_size}  intermediate={model.intermediate_size}  "
          f"experts={model.num_experts}  top_k={model.top_k}  "
          f"block_size={model.block_size}")
    print(f"  batch_size={B}  seq_len={S}  tokens={N}")
    print(f"  M={M}  n_blocks={n_blocks}  padded_tokens={padded_tokens}")

    inputs = torch.randn(B, S, model.hidden_size, dtype=torch.bfloat16)

    # --- Forward pass ---
    if args.simulate:
        print("\nRunning Blockwise Dropless MoE on CPU ...")
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
        print("Running Blockwise Dropless MoE on Trainium ...")
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
