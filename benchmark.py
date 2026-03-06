import argparse
import subprocess
import numpy as np

import torch
import torch_xla.core.xla_model as xm

import nki
#from nki import baremetal
#import neuronxcc.nki as nki
#from neuronxcc.nki import baremetal

from gemm_nki import gemm_nki_kernel
from gemm_nki import matrix_vector_mul_kernel
from naive_moe import routing_kernel, moe_kernel

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

def run_matrix_vector_mul(args):
    """Run and benchmark the matrix-vector multiplication NKI kernel."""
    M, K = 128, 128
    matrix = np.identity(K, dtype=np.float32)  # (M, K)
    vector = np.arange(128, dtype=np.float32)
    #vector = np.ones(K, dtype=np.float32)       # (K,)

    # Kernel expects matT=(K,M) with contraction on P-dim, and vec as 2D (K,1)
    matT = np.ascontiguousarray(matrix.T)        # (K, M)
    vec2d = np.ascontiguousarray(vector.reshape(K, 1))  # (K, 1)

    out = baremetal(matrix_vector_mul_kernel)(matT, vec2d)

    out_ref = matrix.astype(np.float32) @ vector.astype(np.float32)
    out_flat = np.array(out).flatten()
    print(f"Kernel output shape: {np.array(out).shape}")
    print(f"Kernel output: {out_flat[:8]}...")
    if np.allclose(out_flat, out_ref, rtol=1e-2, atol=1e-2):
        print("Correctness: PASSED")
    else:
        max_err = float(np.max(np.abs(out_flat - out_ref)))
        print(f"Correctness: FAILED  (max abs error: {max_err:.6f})")
    
def run_gemm(args):
    """Run and benchmark the GEMM NKI kernel."""
    M, K, N = args.m, args.k, args.n
    dtype = getattr(np, args.dtype)

    print(f"\nGEMM: C({M}x{N}) = A({M}x{K}) @ B({K}x{N})  dtype={args.dtype}")

    A = np.random.rand(M, K).astype(dtype)
    B = np.random.rand(K, N).astype(dtype)

    # Kernel expects lhsT (K, M) with contraction dim first
    A_T = np.ascontiguousarray(A.T)

    # --- Correctness check ---
    if args.check:
        print("Running correctness check against NumPy …")
        if args.simulate:
            C = nki.simulate_kernel(gemm_nki_kernel, A_T, B)
        else:
            C = baremetal(gemm_nki_kernel)(A_T, B)

        C_ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(dtype)
        if np.allclose(C, C_ref, rtol=1e-2, atol=1e-2):
            print("Correctness: PASSED")
        else:
            max_err = float(
                np.max(np.abs(C.astype(np.float32) - C_ref.astype(np.float32)))
            )
            print(f"Correctness: FAILED  (max abs error: {max_err:.6f})")
            return

    if args.simulate:
        print("Benchmark skipped in simulate mode (no Trainium hardware).")
        return

    # --- Benchmark / profile ---
    bench_kwargs = {}
    if args.profile:
        bench_kwargs["save_neff_name"] = args.profile
        bench_kwargs["additional_compile_opt"] = "--disable-dge"

    print(f"Benchmarking (warmup={args.warmup}, iters={args.iters}) …")
    bench_func = nki.benchmark(
        warmup=args.warmup, iters=args.iters, **bench_kwargs
    )(gemm_nki_kernel)
    bench_func(A_T, B)

    p99_us = bench_func.benchmark_result.nc_latency.get_latency_percentile(99)
    print(f"p99 latency: {p99_us:.2f} μs")

    flops = 2 * M * K * N
    tflops = flops / (p99_us * 1e-6) / 1e12
    print(f"Throughput:  {tflops:.2f} TFLOPS")

    if args.profile:
        save_trace(args.profile)

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
    "matrix_vector_mul": run_matrix_vector_mul,
    #"gemm": run_gemm,
    "naive_moe": run_naive_moe_kernel,
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

    # -- GEMM dimensions --
    gemm_group = parser.add_argument_group("GEMM dimensions")
    gemm_group.add_argument(
        "--m", type=int, default=64, help="M (rows of A / C)  [default: 64]",
    )
    gemm_group.add_argument(
        "--k", type=int, default=128, help="K (inner / contraction) [default: 128]",
    )
    gemm_group.add_argument(
        "--n", type=int, default=512, help="N (cols of B / C, multiple of 512)  [default: 512]",
    )
    gemm_group.add_argument(
        "--dtype", type=str, default="float32",
        choices=["float16", "float32"],
        help="Element data type [default: float32]",
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
