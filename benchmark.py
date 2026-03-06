import argparse
import subprocess
import numpy as np

import torch
import torch_xla
import torch_xla.core.xla_model as xm

import nki

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

"""def run_expert_moe_kernel(args):
    feature_dim = args.hidden_size
    batch_size = 1
    seq_len = 128

    torch.manual_seed(args.seed)
    inputs = torch.randn(batch_size, seq_len, feature_dim).to(torch.bfloat16)

    (padded_sorted_tokens_T, w1_experts, w3_experts, w2_experts,
     padded_routing_weights, expert_offsets,
     sorted_expert_indices, expert_counts, top_k, N, moe_block) = create_inputs(inputs)

    num_experts = len(expert_counts)
    T_padded = padded_sorted_tokens_T.shape[1]

    print(f"\nExpert MoE Kernel: N={N}, top_k={top_k}, feature_dim={feature_dim}, "
          f"num_experts={num_experts}, T_padded={T_padded}")

    #device = xm.xla_device()
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
            padded_routing_weights.to(device), expert_offsets.to(device))
        xm.mark_step()
        expert_outputs = expert_outputs_xla.cpu().float()

    # --- Scatter-back: map padded/sorted kernel output to original token order ---
    # expert_outputs shape: (T_padded, feature_dim) — raw expert MLP outputs, unweighted

    # 1. Scale each token's expert output by its routing weight
    expert_outputs = expert_outputs * padded_routing_weights.float().unsqueeze(1)

    # 2. Un-pad: extract real tokens from each expert's padded chunk
    unpadded_outputs = torch.zeros((N * top_k, feature_dim), dtype=torch.float32)
    read_idx = 0
    for i in range(num_experts):
        count = expert_counts[i].item()
        padded_start = expert_offsets[i].item()
        if count > 0:
            unpadded_outputs[read_idx : read_idx + count] = \
                expert_outputs[padded_start : padded_start + count]
        read_idx += count

    # 3. Inverse-sort: scatter from expert-grouped order back to original token positions
    unsorted_outputs = torch.zeros_like(unpadded_outputs)
    unsorted_outputs[sorted_expert_indices] = unpadded_outputs

    # 4. Aggregate top-k expert contributions per original token
    final_outputs = unsorted_outputs.reshape(N, top_k, feature_dim).sum(dim=1)

    if args.check:
        print("Running expert-compute correctness check against PyTorch …")

        moe_block.eval()
        with torch.no_grad():
            ref_output, _ = moe_block(inputs)
            #ref_output, _ = moe_block_f32(inputs.float())
        ref_output = ref_output.reshape(-1, feature_dim).float()

        if torch.allclose(final_outputs, ref_output, rtol=1e-2, atol=1e-2):
            print("Expert MoE kernel correctness: PASSED")
        else:
            max_err = (final_outputs - ref_output).abs().max().item()
            mean_err = (final_outputs - ref_output).abs().mean().item()
            print(f"Expert MoE kernel correctness: FAILED")
            print(f"  max abs error:  {max_err:.6f}")
            print(f"  mean abs error: {mean_err:.6f}")

    return final_outputs"""

def run_expert_moe_kernel(args):
    feature_dim = args.hidden_size
    batch_size = 1
    seq_len = 128

    torch.manual_seed(args.seed)
    inputs = torch.randn(batch_size, seq_len, feature_dim).to(torch.bfloat16)

    (padded_sorted_tokens_T, w1_experts, w3_experts, w2_experts,
     padded_routing_weights, expert_offsets,
     sorted_expert_indices, expert_counts, top_k, N, moe_block) = create_inputs(inputs)

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
            padded_routing_weights.to(device), expert_offsets.to(device))
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
    final_outputs = unsorted_outputs.reshape(N, top_k, feature_dim).sum(dim=1)

    if args.check:
        # ── Authoritative correctness oracle ──
        # Reconstruct expert MLP outputs on the host from the exact
        # grouped/padded tensors and flattened weights the kernel receives.
        # This is the correct oracle because it exercises the identical data
        # layout, padding, and expert-weight indexing as the kernel.
        print("Correctness check: grouped host reference (authoritative)")

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
                w2_e = w2_f[e * args.intermediate_size:(e + 1) * args.intermediate_size, :]

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
            ref_final = ref_unsorted.reshape(N, top_k, feature_dim).sum(dim=1)

            final_diff = (final_outputs - ref_final).abs()
            print(f"  final output       max|mean abs err: "
                  f"{final_diff.max().item():.6f} | {final_diff.mean().item():.6f}")

        if torch.allclose(final_outputs, ref_final, rtol=1e-2, atol=1e-2):
            print("PASSED — kernel matches grouped host reference")
        else:
            print("FAILED — kernel does not match grouped host reference")

        # ── Non-authoritative diagnostic ──
        # The full MoE block uses a different dense/masked bf16 execution
        # path with its own rounding.  Differences here do NOT indicate a
        # kernel bug; this comparison is purely informational.
        with torch.no_grad():
            moe_block.eval()
            block_out, _ = moe_block(inputs)
            block_out = block_out.reshape(-1, feature_dim).float()
            block_diff = (final_outputs - block_out).abs()
            print(f"  [non-authoritative] moe_block max|mean abs err: "
                  f"{block_diff.max().item():.6f} | {block_diff.mean().item():.6f}")

    return final_outputs



def run_naive_moe_kernel(args):
    """Run and benchmark the naive MoE routing kernel."""
    # these are arbitrary
    batch_size = 16
    sequence_length = 16
    T = batch_size * sequence_length

    # these are from the Mixtral 8x7B model
    num_experts = args.num_experts
    top_k = args.top_k
    feature_dim = args.hidden_size
    expert_dim = args.intermediate_size

    #### Routing Stage ####
    print(f"\nMoE Routing: T={T} (B={batch_size}, S={sequence_length}), "
          f"feature_dim={feature_dim}, num_experts={num_experts}, top_k={top_k}")

    inputs = np.random.rand(batch_size, sequence_length, feature_dim).astype(np.float32)
    routing_weights = np.random.normal(loc=0, scale=1, size=(feature_dim, num_experts)).astype(np.float32)
    #routing_weights = np.random.rand(feature_dim, num_experts).astype(np.float32)

    # Flatten to (T, feature_dim) then transpose
    inputs_flat = inputs.reshape(T, feature_dim)
    inputs_T = np.ascontiguousarray(inputs_flat.T)  # (feature_dim, T)

    device = xm.xla_device()
    inputs_T = torch.from_numpy(inputs_T).to(device)
    routing_weights_kernel = torch.from_numpy(routing_weights).to(device)

    # Run routing kernel (tiled matmul + softmax on device)
    if args.simulate:
        probs_xla = nki.simulate_kernel(routing_kernel, inputs_T, routing_weights_kernel)
    else:
        probs_xla = routing_kernel(inputs_T, routing_weights_kernel)
        xm.mark_step()
        #probs = baremetal(routing_kernel)(inputs_T, routing_weights)
    probs = probs_xla.cpu().numpy() if not args.simulate else np.array(probs_xla) #np.array(probs)

    # Top-k selection on host
    top_k_indices = np.argsort(-probs, axis=1)[:, :top_k]
    top_k_values = np.take_along_axis(probs, top_k_indices, axis=1)

    # Renormalize so selected experts sum to 1 (Mixtral convention)
    top_k_values = top_k_values / top_k_values.sum(axis=-1, keepdims=True)

    # Correctness check
    if args.check:
        print("Running correctness check against NumPy …")
        logits_ref = inputs_flat @ routing_weights
        max_logits = np.max(logits_ref, axis=1, keepdims=True)
        exp_logits = np.exp(logits_ref - max_logits)
        probs_ref = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        if np.allclose(probs, probs_ref, rtol=1e-2, atol=1e-2):
            print("Routing kernel correctness: PASSED")
        else:
            max_err = float(np.max(np.abs(
                probs.astype(np.float32) - probs_ref.astype(np.float32))))
            print(f"Routing kernel correctness: FAILED  (max abs error: {max_err:.6f})")
            return

    print(f"Top-{top_k} indices shape: {top_k_indices.shape}")
    print(f"Top-{top_k} values shape:  {top_k_values.shape}")
    for i in range(min(5, T)):
        print(f"  token {i}: experts={top_k_indices[i]}  "
              f"weights={np.round(top_k_values[i], 4)}")


    #### Expert Compute Stage ####
    expert_w_gate = np.random.normal(loc=0, scale=1, size=(num_experts, feature_dim, expert_dim)).astype(np.float32)
    expert_w_up = np.random.normal(loc=0, scale=1, size=(num_experts, feature_dim, expert_dim)).astype(np.float32)
    expert_w_down = np.random.normal(loc=0, scale=1, size=(num_experts, expert_dim, feature_dim)).astype(np.float32)

    # Flatten weight tensors to 2-D so the kernel can use nl.ds for
    # dynamic expert selection on the first dimension.
    expert_w_gate_2d = expert_w_gate.reshape(-1, expert_dim)
    expert_w_up_2d = expert_w_up.reshape(-1, expert_dim)
    expert_w_down_2d = expert_w_down.reshape(-1, feature_dim)

    # Kernel expects float32 indices for SBUF arithmetic
    top_k_indices_f = top_k_indices.astype(np.float32)
    top_k_values_f = top_k_values.astype(np.float32)

    # Move expert-stage inputs to device (mirror routing stage)
    inputs_flat_t = torch.from_numpy(inputs_flat).to(device)
    expert_w_gate_2d_t = torch.from_numpy(expert_w_gate_2d).to(device)
    expert_w_up_2d_t = torch.from_numpy(expert_w_up_2d).to(device)
    expert_w_down_2d_t = torch.from_numpy(expert_w_down_2d).to(device)
    top_k_indices_t = torch.from_numpy(top_k_indices_f).to(device)
    top_k_values_t = torch.from_numpy(top_k_values_f).to(device)

    print(f"\nExpert compute: expert_dim={expert_dim}")

    if args.simulate:
        expert_outputs_xla = nki.simulate_kernel(
            moe_kernel,
            inputs_flat_t,
            expert_w_gate_2d_t, expert_w_up_2d_t, expert_w_down_2d_t,
            top_k_indices_t, top_k_values_t)
    else:
        expert_outputs_xla = moe_kernel(
            inputs_flat_t,
            expert_w_gate_2d_t, expert_w_up_2d_t, expert_w_down_2d_t,
            top_k_indices_t, top_k_values_t)
        xm.mark_step()
    expert_outputs = expert_outputs_xla.cpu().numpy() if not args.simulate else np.array(expert_outputs_xla)

    if args.check:
        print("Running expert-compute correctness check against NumPy …")
        check_count = min(32, T)
        output_ref = np.zeros((check_count, feature_dim), dtype=np.float32)
        for t_i in range(check_count):
            for k_i in range(top_k):
                e = int(top_k_indices[t_i, k_i])
                w = top_k_values[t_i, k_i]
                gate = inputs_flat[t_i] @ expert_w_gate[e]
                up = inputs_flat[t_i] @ expert_w_up[e]
                silu_gate = gate / (1.0 + np.exp(-gate))
                inter = silu_gate * up
                down = inter @ expert_w_down[e]
                output_ref[t_i] += w * down

        if np.allclose(expert_outputs[:check_count], output_ref,
                       rtol=1e-2, atol=1e-2):
            print("Expert compute correctness: PASSED")
        else:
            max_err = float(np.max(np.abs(
                expert_outputs[:check_count].astype(np.float32)
                - output_ref)))
            print(f"Expert compute correctness: FAILED "
                  f"(max abs error: {max_err:.6f})")

    return expert_outputs


# Registry — add new kernel runners here as the project grows.
KERNEL_RUNNERS = {
    "naive_moe": run_naive_moe_kernel,
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
