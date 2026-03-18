#!/usr/bin/env python3
"""
Pure-PyTorch MoE layer on Trainium2 (no custom NKI kernel).

Uses the MixtralSparseMoeBlock directly via torch_xla.  This serves as the
baseline comparison against the NKI-kernel version in moe_nki.py.

Usage:
    # Hardware run with correctness check
    python moe_pytorch.py --batch-size 1 --seq-len 128 --check

    # Larger workload with profiling
    python moe_pytorch.py --batch-size 1 --seq-len 1024 --profile moe_pytorch
"""

import argparse
import os
import subprocess

import torch
import torch.nn as nn

from mistral_moe import MixtralConfig, MixtralSparseMoeBlock

MODEL_PATH = "weights/uncompiled_model_weights/mistral_moe.pt"


def check_correctness(tag, actual, expected, rtol=5e-2, atol=5e-2):
    """Print max/mean abs error and PASS/FAIL."""
    diff = (actual.float() - expected.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    print(f"  [{tag}] max|mean abs err: {max_err:.6f} | {mean_err:.6f}")
    ok = torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol)
    print(f"  [{tag}] {'PASSED' if ok else 'FAILED'}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Pure-PyTorch MoE on Trainium2 (baseline, no NKI kernel)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size [default: 1]")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length [default: 128]")
    parser.add_argument("--check", action="store_true",
                        help="Correctness check vs CPU reference")
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
    config = MixtralConfig()

    print("=" * 60)
    print("PyTorch MoE Layer (baseline) — Trainium2")
    print("=" * 60)
    print(f"batch_size={B}  seq_len={S}  tokens={N}")
    print(f"  hidden={config.hidden_size}  intermediate={config.intermediate_size}  "
          f"experts={config.num_local_experts}  top_k={config.num_experts_per_tok}")

    # ── Build model ──
    model = MixtralSparseMoeBlock(config)

    use_pretrained = (not args.random_weights) and os.path.exists(MODEL_PATH)
    if use_pretrained:
        state_dict = torch.load(MODEL_PATH, map_location="cpu",
                                weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {MODEL_PATH}")
    else:
        if not args.random_weights and not os.path.exists(MODEL_PATH):
            print(f"WARNING: {MODEL_PATH} not found — using random weights")
        print("Using random weights")

    model = model.to(dtype=torch.bfloat16)
    model.eval()

    inputs = torch.randn(B, S, config.hidden_size, dtype=torch.bfloat16)

    # ── CPU reference (before moving to device) ──
    cpu_out = None
    if args.check:
        print("\nComputing CPU reference ...")
        with torch.no_grad():
            cpu_out, _ = model(inputs)
        cpu_out = cpu_out.float()

    # ── Hardware forward pass ──
    import torch_xla
    import torch_xla.core.xla_model as xm

    device = torch_xla.device()
    model = model.to(device)
    inputs_dev = inputs.to(device)

    print(f"\nRunning PyTorch MoE on Trainium ...")
    with torch.no_grad():
        output, selected_experts = model(inputs_dev)
    xm.mark_step()
    output_f32 = output.cpu().float()
    selected_experts_cpu = selected_experts.cpu()

    print(f"Output shape: {list(output_f32.shape)}")
    print(f"Output range: [{output_f32.min().item():.4f}, "
          f"{output_f32.max().item():.4f}]")

    # ── Correctness ──
    if args.check and cpu_out is not None:
        print("\n--- Correctness Check ---")
        check_correctness("vs CPU reference (independent routing)",
                          output_f32, cpu_out)

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
