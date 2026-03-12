import argparse
import subprocess
import numpy as np

import torch
import torch_xla
import torch_xla.core.xla_model as xm

import nki
import nki.language as nl
import nki.isa as nisa

from naive_moe import routing_kernel, moe_kernel, moe_expert_kernel
from host import create_inputs
from mistral_moe import MixtralConfig, MixtralSparseMoeBlock

def save_trace(profile_name):
    """Run neuron-profile to capture NEFF/NTFF trace files."""
    subprocess.run(
        [
            "neuron-profile", "capture",
            "-n", f"{profile_name}.neff",
            "-s", f"{profile_name}.ntff",
        ],
        check=True,
    )
    print(f"Trace saved: {profile_name}.neff, {profile_name}.ntff")


def correctness_check(
    expert_outputs, final_outputs, inputs,
    padded_sorted_tokens_T, w1_experts, w3_experts, w2_experts,
    padded_routing_weights, expert_offsets, expert_counts,
    sorted_expert_indices, moe_block,
    feature_dim, intermediate_size, num_experts, T_padded, N, top_k,
):
    """Compare kernel output against a host BF16 CPU reference AND the Mistral MoE block.

    Returns (ref_host_final, ref_block_final) — both are (N, feature_dim) float tensors.
    """
    print("Correctness check: Host reference")

    with torch.no_grad():
        x_padded = padded_sorted_tokens_T.T.float().contiguous()
        w1_f = w1_experts.float().contiguous()
        w3_f = w3_experts.float().contiguous()
        w2_f = w2_experts.float().contiguous()

        ref_raw = torch.zeros((T_padded, feature_dim), dtype=torch.float32)

        for e in range(num_experts):
            start = expert_offsets[e].item()
            end = expert_offsets[e + 1].item()
            count = expert_counts[e].item()

            x_chunk = x_padded[start:end]
            w1_e = w1_f[e * feature_dim:(e + 1) * feature_dim, :]
            w3_e = w3_f[e * feature_dim:(e + 1) * feature_dim, :]
            w2_e = w2_f[e * intermediate_size:(e + 1) * intermediate_size, :]

            gate = x_chunk @ w1_e
            up = x_chunk @ w3_e
            inter = torch.nn.functional.silu(gate) * up
            ref_raw[start:end] = inter @ w2_e

            if count < (end - start):
                ref_raw[start + count:end].zero_()

        raw_diff = (expert_outputs - ref_raw).abs()
        print(f"  raw expert output  max|mean abs err: "
              f"{raw_diff.max().item():.6f} | {raw_diff.mean().item():.6f}")

        ref_weighted = ref_raw * padded_routing_weights.float().unsqueeze(1)

        ref_unpadded = torch.zeros((N * top_k, feature_dim), dtype=torch.float32)
        read_idx = 0
        for i in range(num_experts):
            count = expert_counts[i].item()
            padded_start = expert_offsets[i].item()
            if count > 0:
                ref_unpadded[read_idx:read_idx + count] = \
                    ref_weighted[padded_start:padded_start + count]
            read_idx += count

        ref_unsorted = torch.zeros_like(ref_unpadded)
        ref_unsorted[sorted_expert_indices] = ref_unpadded
        ref_host_final = ref_unsorted.reshape(N, top_k, feature_dim).sum(dim=1)

        final_diff = (final_outputs - ref_host_final).abs()
        print(f"  final output       max|mean abs err: "
              f"{final_diff.max().item():.6f} | {final_diff.mean().item():.6f}")

    if torch.allclose(final_outputs, ref_host_final, rtol=2e-2, atol=2e-2):
        print("PASSED — kernel matches host reference")
    else:
        print("FAILED — kernel does not match grouped host reference")

    with torch.no_grad():
        moe_block.eval()
        block_out, _ = moe_block(inputs)
        ref_block_final = block_out.reshape(-1, feature_dim).float()
        block_diff = (final_outputs - ref_block_final).abs()
        print(f"  [non-authoritative] moe_block max|mean abs err: "
              f"{block_diff.max().item():.6f} | {block_diff.mean().item():.6f}")

    return ref_host_final, ref_block_final


def run_expert_moe_kernel(args):
    feature_dim = args.hidden_size
    batch_size = 1
    seq_len = 128

    torch.manual_seed(args.seed)
    inputs = torch.randn(batch_size, seq_len, feature_dim).to(torch.bfloat16)

    (padded_sorted_tokens_T, w1_experts, w3_experts, 
    w2_experts, padded_routing_weights, expert_offsets, 
    sorted_expert_indices, expert_counts, top_k, N, moe_block) = create_inputs(inputs)

    assert N == batch_size * seq_len, f"N != num tokens. N={N}, tokens = {batch_size * seq_len}"

    print(f"padded_sorted_tokens_T shape: {padded_sorted_tokens_T.shape}")

    num_experts = len(expert_counts)
    T_padded = padded_sorted_tokens_T.shape[1]

    print(f"\nExpert MoE Kernel: N={N}, top_k={top_k}, feature_dim={feature_dim}, "
          f"num_experts={num_experts}, T_padded={T_padded}")

    device = torch_xla.device()

    if args.simulate:
        expert_outputs_xla = nki.simulate_kernel(
            moe_expert_kernel,
            padded_sorted_tokens_T,
            w1_experts, w3_experts, w2_experts,
            padded_routing_weights, expert_offsets)
        expert_outputs = torch.from_numpy(np.array(expert_outputs_xla)).float()
    else:
        expert_outputs_xla = moe_expert_kernel(
            padded_sorted_tokens_T.to(device),
            w1_experts.to(device), w3_experts.to(device), w2_experts.to(device), 
            expert_offsets.to(device))
        xm.mark_step()
        expert_outputs = expert_outputs_xla.cpu().float()

    # Scatter-back: routing-weight multiply → unpad → inverse-sort → top-k sum
    expert_outputs_weighted = expert_outputs * padded_routing_weights.float().unsqueeze(1)

    unpadded_outputs = torch.zeros((N * top_k, feature_dim), dtype=torch.float32)
    read_idx = 0
    for i in range(num_experts):
        count = expert_counts[i].item()
        padded_start = expert_offsets[i].item()
        if count > 0:
            unpadded_outputs[read_idx:read_idx + count] = \
                expert_outputs_weighted[padded_start:padded_start + count]
        read_idx += count

    unsorted_outputs = torch.zeros_like(unpadded_outputs)
    unsorted_outputs[sorted_expert_indices] = unpadded_outputs

    # kernel outputs weighted, summed, and reordered:
    final_outputs = unsorted_outputs.reshape(N, top_k, feature_dim).sum(dim=1)

    if args.check:
        ref_host, ref_block = correctness_check(
            expert_outputs, final_outputs, inputs,
            padded_sorted_tokens_T, w1_experts, w3_experts, w2_experts,
            padded_routing_weights, expert_offsets, expert_counts,
            sorted_expert_indices, moe_block,
            feature_dim, args.intermediate_size, num_experts, T_padded, N, top_k,
        )

    return final_outputs

    


# Registry — add new kernel runners here as the project grows.
KERNEL_RUNNERS = {
    "expert_moe": run_expert_moe_kernel,
}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and profile MoE kernels on Trainium2",
    )

    parser.add_argument(
        "--kernel", type=str, default="matrix_vector_mul",
        choices=list(KERNEL_RUNNERS.keys()),
        help="Kernel to benchmark (default: matrix_vector_mul)",
    )
    parser.add_argument(
        "--profile", type=str, default=None, metavar="NAME",
        help="Profile the run and save trace files as NAME.neff / NAME.ntff",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run on CPU via NKI simulator (no Trainium hardware required)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Verify kernel correctness against a NumPy reference",
    )


    # -- MoE configuration (reserved for future kernels) --
    moe_group = parser.add_argument_group("MoE configuration (future use)")
    moe_group.add_argument(
        "--num-experts", type=int, default=8,
        help="Number of experts [default: 8]",
    )
    moe_group.add_argument(
        "--top-k", type=int, default=2,
        help="Top-k experts per token [default: 2]",
    )
    moe_group.add_argument(
        "--hidden-size", type=int, default=4096,
        help="Hidden dimension [default: 4096]",
    )
    moe_group.add_argument(
        "--intermediate-size", type=int, default=14336,
        help="Intermediate (expert FFN) dimension [default: 14336]",
    )

    # -- Benchmark tuning --
    bench_group = parser.add_argument_group("Benchmark tuning")
    bench_group.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations [default: 5]",
    )
    bench_group.add_argument(
        "--iters", type=int, default=20, help="Timed iterations [default: 20]",
    )
    bench_group.add_argument(
        "--seed", type=int, default=42, help="Random seed [default: 42]",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    print("=" * 50)
    print("MoE Kernel Benchmark — Trainium2")
    print("=" * 50)
    print(f"Kernel:  {args.kernel}")
    print(f"Mode:    {'simulate' if args.simulate else 'hardware'}")
    if args.profile:
        print(f"Profile: {args.profile}")

    runner = KERNEL_RUNNERS.get(args.kernel)
    if runner:
        runner(args)
    else:
        parser.error(f"No runner registered for kernel '{args.kernel}'")


if __name__ == "__main__":
    main()
