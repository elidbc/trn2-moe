#!/usr/bin/env python3
"""
End-to-end MoE layer on Trainium2 using a custom NKI expert kernel.

Routing (token->expert dispatch) and recombination (weighted top-k sum) are
in PyTorch; the expert FFN compute uses moe_expert_kernel from naive_moe.py.

Running on hardware automatically generates neff files in the Neuron compile
cache.  Use --profile to capture ntff trace files for neuron-profile.

Usage:
    # Hardware run with correctness check
    python moe_nki.py --batch-size 1 --seq-len 128 --check

    # Pure-PyTorch simulation (no Trainium required)
    python moe_nki.py --batch-size 1 --seq-len 128 --check --simulate

    # Larger workload with profiling
    python moe_nki.py --batch-size 1 --seq-len 1024 --profile moe_full
"""

import argparse
import math
import os
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from naive_moe import moe_expert_kernel
from mistral_moe import MixtralConfig, MixtralSparseMoeBlock

TILE_SIZE = 128
MODEL_PATH = "weights/uncompiled_model_weights/mistral_moe.pt"


def _max_tok_per_expert(num_tokens, top_k, num_experts):
    """Padded max tokens per expert bin (1x expected, tile-aligned, min 1 tile)."""
    expected = (num_tokens * top_k) / num_experts
    return max(math.ceil(expected / TILE_SIZE) * TILE_SIZE, TILE_SIZE)


class NKIMoELayer(nn.Module):
    """Full MoE layer: router -> sort/pad -> NKI expert kernel -> recombine.

    The within-expert placement is computed via a one-hot cumulative sum,
    keeping all tensor shapes static for XLA compilation.  Expert bins are
    padded to a fixed ``max_tok_per_expert`` (the expected count, rounded
    up to TILE_SIZE).  Padding tokens produce zero output and are harmless.

    ``forward()`` returns ``(output, topk_idx, routing_weights)`` so that
    correctness checks can reuse the device's exact routing decisions.
    """

    def __init__(self, hidden_size=4096, intermediate_size=14336,
                 num_experts=8, top_k=2, simulate=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.simulate = simulate

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

    # ── Factory helpers ──────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, model_path=MODEL_PATH, simulate=False):
        """Load weights from a saved MixtralSparseMoeBlock state dict.

        Returns ``(NKIMoELayer, MixtralSparseMoeBlock)`` so the original
        block can be used as a secondary correctness reference.
        """
        config = MixtralConfig()
        layer = cls(
            config.hidden_size, config.intermediate_size,
            config.num_local_experts, config.num_experts_per_tok,
            simulate=simulate,
        )

        state_dict = torch.load(model_path, map_location="cpu",
                                weights_only=True)
        block = MixtralSparseMoeBlock(config)
        block.load_state_dict(state_dict)
        block = block.to(dtype=torch.bfloat16)
        block.eval()

        layer.router.weight.data.copy_(block.router.router.weight.data)

        w1, w2, w3 = [], [], []
        for expert in block.experts:
            w1.append(expert.w1.weight.data.T)  # [out,in] -> [in,out]
            w2.append(expert.w2.weight.data.T)
            w3.append(expert.w3.weight.data.T)

        layer.w1_experts.copy_(
            torch.stack(w1).reshape(-1, config.intermediate_size))
        layer.w3_experts.copy_(
            torch.stack(w3).reshape(-1, config.intermediate_size))
        layer.w2_experts.copy_(
            torch.stack(w2).reshape(-1, config.hidden_size))

        return layer.to(dtype=torch.bfloat16), block

    @classmethod
    def random_init(cls, hidden_size=4096, intermediate_size=14336,
                    num_experts=8, top_k=2, simulate=False):
        """Create with Kaiming-uniform random weights."""
        layer = cls(hidden_size, intermediate_size, num_experts, top_k,
                    simulate=simulate)
        nn.init.kaiming_uniform_(layer.router.weight)
        nn.init.kaiming_uniform_(layer.w1_experts)
        nn.init.kaiming_uniform_(layer.w3_experts)
        nn.init.kaiming_uniform_(layer.w2_experts)
        return layer.to(dtype=torch.bfloat16)

    # ── Forward ──────────────────────────────────────────────────────

    def forward(self, hidden_states):
        """Run the full MoE layer.

        Returns:
            (output, topk_idx, routing_weights) where topk_idx is (N, top_k)
            expert indices and routing_weights is (N, top_k) softmax weights.
            Returning routing info lets correctness checks use the exact same
            routing decisions the device produced.
        """
        B, S, D = hidden_states.shape
        N = B * S
        x = hidden_states.reshape(N, D)

        # ── Router ──
        logits = self.router(x)
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        rw = F.softmax(topk_vals, dim=-1, dtype=torch.float32).to(x.dtype)

        # ── Flatten for top-k processing ──
        flat_exp = topk_idx.reshape(-1).long()            # (N*k,)
        flat_wt = rw.reshape(-1)                          # (N*k,)
        repeated = (x.unsqueeze(1)
                     .expand(-1, self.top_k, -1)
                     .reshape(-1, D))                     # (N*k, D)

        # ── Within-expert rank via one-hot cumulative sum ──
        max_tok = _max_tok_per_expert(N, self.top_k, self.num_experts)
        T_pad = self.num_experts * max_tok

        oh = F.one_hot(flat_exp, self.num_experts).to(torch.float32)
        rank = (torch.cumsum(oh, dim=0) * oh).sum(-1).long() - 1
        rank = rank.clamp(max=max_tok - 1)
        dest = flat_exp * max_tok + rank                  # (N*k,)

        # ── Scatter tokens and routing weights into padded expert bins ──
        pad_tok = torch.zeros(T_pad, D, dtype=x.dtype, device=x.device)
        pad_tok.scatter_(0, dest.unsqueeze(1).expand(-1, D), repeated)

        pad_wt = torch.zeros(T_pad, dtype=x.dtype, device=x.device)
        pad_wt.scatter_(0, dest, flat_wt)

        # ── Expert FFN ──
        if self.simulate:
            expert_out = self._pytorch_expert_ffn(pad_tok, max_tok)
        else:
            pad_tok_T = pad_tok.T.contiguous()            # (D, T_pad)
            offsets = (torch.arange(self.num_experts + 1,
                                    dtype=torch.int32,
                                    device=x.device) * max_tok)
            expert_out = moe_expert_kernel(
                pad_tok_T,
                self.w1_experts, self.w3_experts, self.w2_experts,
                offsets,
            )

        # ── Recombine: route-weight -> gather -> top-k sum ──
        weighted = expert_out.float() * pad_wt.float().unsqueeze(1)
        gathered = torch.gather(
            weighted, 0, dest.unsqueeze(1).expand(-1, D))
        final = gathered.reshape(N, self.top_k, D).sum(dim=1)

        output = final.to(x.dtype).reshape(B, S, D)
        return output, topk_idx, rw

    def _pytorch_expert_ffn(self, pad_tok, max_tok):
        """Pure-PyTorch expert FFN fallback (replaces NKI kernel for --simulate).

        Matches the kernel's computation: for each expert's token chunk,
        compute SiLU(x @ W1) * (x @ W3) then @ W2.  Accumulates in float32,
        returns in the input dtype (matching the NKI kernel's behaviour).
        """
        T_pad, D = pad_tok.shape
        result = torch.zeros(T_pad, D, dtype=torch.float32,
                             device=pad_tok.device)
        for e in range(self.num_experts):
            start = e * max_tok
            end = (e + 1) * max_tok
            tok = pad_tok[start:end].float()
            w1_e = self.w1_experts[e * D:(e + 1) * D].float()
            w3_e = self.w3_experts[e * D:(e + 1) * D].float()
            w2_e = self.w2_experts[
                e * self.intermediate_size:(e + 1) * self.intermediate_size
            ].float()
            result[start:end] = (F.silu(tok @ w1_e) * (tok @ w3_e)) @ w2_e
        return result.to(pad_tok.dtype)


# ── CPU reference for correctness ────────────────────────────────────

@torch.no_grad()
def cpu_reference(inputs_bf16, w1, w3, w2,
                  num_experts, top_k, hidden_size, intermediate_size,
                  topk_idx, rw):
    """Float32 CPU reference using the caller-provided routing decisions.

    By accepting ``topk_idx`` and ``rw`` from the device run we eliminate
    routing divergence (bf16 arithmetic can pick different top-k experts on
    Trainium vs CPU) and isolate the check to expert FFN + recombination.
    """
    B, S, D = inputs_bf16.shape
    N = B * S
    x = inputs_bf16.reshape(N, D)

    flat_exp = topk_idx.reshape(-1).long()
    flat_wt = rw.float().reshape(-1)
    repeated = x.float().unsqueeze(1).expand(-1, top_k, -1).reshape(-1, D)

    result = torch.zeros(N * top_k, D, dtype=torch.float32)
    for e in range(num_experts):
        mask = flat_exp == e
        if not mask.any():
            continue
        tok = repeated[mask]
        w1_e = w1[e * D:(e + 1) * D].float()
        w3_e = w3[e * D:(e + 1) * D].float()
        w2_e = w2[e * intermediate_size:(e + 1) * intermediate_size].float()
        result[mask] = (F.silu(tok @ w1_e) * (tok @ w3_e)) @ w2_e

    weighted = result * flat_wt.unsqueeze(1)
    return weighted.reshape(N, top_k, D).sum(dim=1).reshape(B, S, D)


def check_correctness(tag, actual, expected, rtol=5e-2, atol=5e-2):
    """Print max/mean abs error and PASS/FAIL."""
    diff = (actual.float() - expected.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    print(f"  [{tag}] max|mean abs err: {max_err:.6f} | {mean_err:.6f}")
    ok = torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol)
    print(f"  [{tag}] {'PASSED' if ok else 'FAILED'}")
    return ok


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end MoE on Trainium2 with custom NKI kernel")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size [default: 1]")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length [default: 128]")
    parser.add_argument("--simulate", action="store_true",
                        help="Pure-PyTorch fallback (no Trainium needed)")
    parser.add_argument("--check", action="store_true",
                        help="Correctness check vs CPU float32 reference")
    parser.add_argument("--profile", type=str, default=None, metavar="NAME",
                        help="Capture profile as NAME.neff / NAME.ntff")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed [default: 42]")
    parser.add_argument("--random-weights", action="store_true",
                        help="Random weights (skip pretrained model)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    B, S = args.batch_size, args.seq_len
    N = B * S

    print("=" * 60)
    print("NKI MoE Layer — Trainium2")
    print("=" * 60)
    print(f"batch_size={B}  seq_len={S}  tokens={N}")
    print(f"mode: {'simulate' if args.simulate else 'hardware'}")

    # ── Build model ──
    ref_block = None
    use_pretrained = (not args.random_weights) and os.path.exists(MODEL_PATH)

    if use_pretrained:
        model, ref_block = NKIMoELayer.from_pretrained(
            simulate=args.simulate)
        print(f"Loaded pretrained weights from {MODEL_PATH}")
    else:
        if not args.random_weights and not os.path.exists(MODEL_PATH):
            print(f"WARNING: {MODEL_PATH} not found — using random weights")
        model = NKIMoELayer.random_init(simulate=args.simulate)
        print("Initialized with random weights")

    print(f"  hidden={model.hidden_size}  intermediate={model.intermediate_size}  "
          f"experts={model.num_experts}  top_k={model.top_k}")

    max_tok = _max_tok_per_expert(N, model.top_k, model.num_experts)
    T_pad = model.num_experts * max_tok
    print(f"  max_tok_per_expert={max_tok}  T_padded={T_pad}")

    inputs = torch.randn(B, S, model.hidden_size, dtype=torch.bfloat16)

    # Save CPU copies of expert weights before potential device move
    w1_cpu = model.w1_experts.clone()
    w3_cpu = model.w3_experts.clone()
    w2_cpu = model.w2_experts.clone()

    # ── Forward pass ──
    if args.simulate:
        print("\nRunning NKI MoE (PyTorch simulate) ...")
        with torch.no_grad():
            output, topk_idx, rw = model(inputs)
        output_f32 = output.float()
        topk_idx_cpu = topk_idx
        rw_cpu = rw
    else:
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = torch_xla.device()
        model = model.to(device)
        inputs_dev = inputs.to(device)

        print("\nRunning NKI MoE on Trainium ...")
        with torch.no_grad():
            output, topk_idx, rw = model(inputs_dev)
        xm.mark_step()
        output_f32 = output.cpu().float()
        topk_idx_cpu = topk_idx.cpu()
        rw_cpu = rw.cpu()

    print(f"Output shape: {list(output_f32.shape)}")
    print(f"Output range: [{output_f32.min().item():.4f}, "
          f"{output_f32.max().item():.4f}]")

    # ── Correctness ──
    if args.check:
        print("\n--- Correctness Check ---")

        ref_out = cpu_reference(
            inputs, w1_cpu, w3_cpu, w2_cpu,
            model.num_experts, model.top_k,
            model.hidden_size, model.intermediate_size,
            topk_idx_cpu, rw_cpu,
        )
        check_correctness("vs cpu_reference (matched routing)", output_f32,
                          ref_out)

        if ref_block is not None:
            with torch.no_grad():
                block_out, _ = ref_block(inputs.cpu().to(torch.bfloat16))
            check_correctness(
                "vs MixtralSparseMoeBlock (independent routing)",
                output_f32,
                block_out.reshape_as(output_f32).float(),
            )

    # ── Profile ──
    """if args.profile:
        print(f"\nCapturing trace -> {args.profile}.neff / {args.profile}.ntff")
        subprocess.run(
            ["neuron-profile", "capture",
             "-n", f"{args.profile}.neff",
             "-s", f"{args.profile}.ntff"],
            check=True,
        )"""

    print("\nDone.")


if __name__ == "__main__":
    main()
