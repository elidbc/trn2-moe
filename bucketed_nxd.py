#!/usr/bin/env python3
"""
Bucketed MoE Layer (neuronx_distributed) — Benchmark on Trainium2

Uses the NeuronX Distributed (nxd) MoE module with blockwise matmul
for efficient expert computation on Trainium hardware.

Usage:
    python bucketed_nxd.py --batch-size 1 --seq-len 128
    python bucketed_nxd.py --batch-size 4 --seq-len 512
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist


def init_distributed():
    """Bootstrap torch.distributed for single-process nxd usage."""
    if dist.is_initialized():
        return
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('LOCAL_RANK', '0')

    import torch_xla.distributed.xla_backend  # noqa: F401  (registers 'xla' backend)
    dist.init_process_group(backend='xla', init_method='xla://')


def init_parallel():
    from neuronx_distributed.parallel_layers import parallel_state

    init_distributed()
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
        )


def build_moe(
    hidden_size=4096,
    intermediate_size=14336,
    num_experts=8,
    top_k=2,
    block_size=512,
):
    from neuronx_distributed.modules.moe import MoE, routing
    from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
    from neuronx_distributed.modules.moe.moe_configs import (
        RoutedExpertsMLPOpsConfig,
        BlockwiseMatmulConfig,
    )

    init_parallel()

    router = routing.RouterTopK(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        act_fn="softmax",
        sequence_parallel_enabled=False,
    )

    routed_cfg = RoutedExpertsMLPOpsConfig(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        glu_mlp=True,
        capacity_factor=None,
        normalize_top_k_affinities=True,
        use_index_calc_kernel=True,
        is_prefill=True,
    )

    block_cfg = BlockwiseMatmulConfig.from_kwargs(
        block_size=block_size,
        logical_nc_config=2,
        skip_dma_token=True,
        skip_dma_weight=True,
        use_shard_on_intermediate_dynamic_while=True,
    )

    expert_mlps = ExpertMLPsV2(
        routed_experts_mlp_config=routed_cfg,
        blockwise_matmul_config=block_cfg,
        sequence_parallel_enabled=False,
    )

    moe = MoE(
        router=router,
        expert_mlps=expert_mlps,
        sequence_parallel_enabled=False,
    )
    return moe


def cast_module_bf16(module: nn.Module):
    for _, p in module.named_parameters(recurse=True):
        if p.is_floating_point():
            p.data = p.data.to(torch.bfloat16)
    for _, b in module.named_buffers(recurse=True):
        if torch.is_tensor(b) and b.is_floating_point():
            b.data = b.data.to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(
        description="Bucketed MoE (nxd) — benchmark on Trainium2")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size [default: 1]")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length [default: 128]")
    parser.add_argument("--hidden-size", type=int, default=4096,
                        help="Hidden dimension [default: 4096]")
    parser.add_argument("--intermediate-size", type=int, default=14336,
                        help="Intermediate dimension [default: 14336]")
    parser.add_argument("--num-experts", type=int, default=8,
                        help="Number of experts [default: 8]")
    parser.add_argument("--top-k", type=int, default=2,
                        help="Top-k routing [default: 2]")
    parser.add_argument("--block-size", type=int, default=512,
                        help="Blockwise matmul block size [default: 512]")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed [default: 42]")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    B, S = args.batch_size, args.seq_len

    print("=" * 60)
    print("Bucketed MoE (nxd) — Trainium2 Benchmark")
    print("=" * 60)

    import torch_xla

    device = torch_xla.device()
    print(f"device: {device}")

    print("Using random weights")
    moe = build_moe(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        block_size=args.block_size,
    ).to(device).eval()

    cast_module_bf16(moe)

    print(f"  hidden={args.hidden_size}  intermediate={args.intermediate_size}  "
          f"experts={args.num_experts}  top_k={args.top_k}  "
          f"block_size={args.block_size}")
    print(f"  batch_size={B}  seq_len={S}  tokens={B * S}")

    x = torch.randn(B, S, args.hidden_size, dtype=torch.bfloat16, device=device)

    # Warmup (triggers XLA compilation)
    print("\nWarming up (XLA compile) ...")
    with torch.no_grad():
        _ = moe(x)
    torch_xla.sync()

    # Timed run
    print("Running Bucketed MoE (nxd) on Trainium ...")
    with torch.no_grad():
        t0 = time.perf_counter()
        y = moe(x)
        torch_xla.sync()
        t1 = time.perf_counter()

    out = y[0] if isinstance(y, tuple) else y
    output = out.cpu()

    print(f"  Elapsed: {(t1 - t0) * 1000:.2f} ms")
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Output range: [{output.float().min().item():.4f}, "
          f"{output.float().max().item():.4f}]")
    print("\nDone.")


if __name__ == "__main__":
    main()
