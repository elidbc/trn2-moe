# MoE Optimization Benchmark — Experiment Guide

## Overview

This benchmark measures how different Neuron Distributed (NxD) optimizations affect the performance of a **Mixture of Experts (MoE)** layer from **Mistral 8x7B** running on **AWS Trainium (trn2)**. The goal is to isolate the effect of each optimization so you can make clean performance claims.

**Hardware**: trn2.3xlarge — 1 Neuron device, 4 NeuronCores (LNC=2), 96 GB HBM  
**Model layer**: Mistral 8x7B MoE block — 8 experts, top-2 routing, hidden=4096, intermediate=14336  
**Framework**: NxD (neuronx-distributed) with PyTorch XLA

---

## Key Concepts

### Mixture of Experts (MoE)
Instead of one large feed-forward layer, MoE uses multiple smaller "expert" networks. A **router** selects the top-*k* experts for each token. Only the selected experts process each token, so the compute cost scales with *k* (not the total expert count). Mistral 8x7B uses 8 experts with top-2 routing.

### Tensor Parallelism (TP)
Splits each expert's weight matrices **across NeuronCores**. Every core holds a slice of every expert. Requires **all-reduce** communication after each layer to combine results. Higher TP = smaller per-core memory, more communication.

### Expert Parallelism (EP)
Assigns **different experts to different NeuronCores**. Each core holds full copies of a subset of experts. Requires **all-to-all** communication to route tokens to the right core. Higher EP = each core specializes in fewer experts.

### Sequence Parallelism (SP)
Distributes **activation memory along the sequence dimension** across TP workers. Adds an all-gather before the MoE layer and a reduce-scatter after. Reduces peak memory at the cost of extra communication. Only meaningful when TP > 1.

### Blockwise Matrix Multiplication (BWMM)
NxD's optimized kernel for MoE computation. Instead of running each expert as a separate matmul, BWMM:
- Groups tokens into fixed-size **blocks** (default 512 tokens)
- Assigns each block to an expert
- Runs all blocks as a single fused operation
- Uses Neuron Kernel Interface (NKI) for hardware-level optimization
- Includes **DMA skip** optimizations for token and weight transfers

### Token Dropping (Capacity Factor)
With `capacity_factor=None` (dropless), every token is processed regardless of load imbalance. With `capacity_factor=1.0`, each expert can process at most `(total_tokens × top_k × CF) / num_experts` tokens. Excess tokens are dropped. Dropping guarantees bounded compute but may hurt accuracy.

### Input Distribution
- **Uniform**: Tokens are evenly distributed across all 8 experts. Represents ideal load balancing.
- **Skewed**: >90% of tokens route to experts 0 and 1. Represents worst-case load imbalance (common in practice with popular topics/domains). Generated via gradient descent against the actual router weights.

---

## Experiment Structure

### Design Principle
Each experiment group changes **exactly one variable** from a shared baseline configuration. This lets you attribute performance differences to that single variable.

### Group 1: Single NeuronCore — BWMM Effect (nproc=1)

| Config | TP | EP | BWMM | SP | CF | What it shows |
|--------|----|----|------|----|----|---------------|
| `single_baseline` | 1 | 1 | OFF | OFF | — | **Baseline**: no optimizations |
| `single_bwmm` | 1 | 1 | **ON** | OFF | — | **BWMM speedup** over baseline |
| `single_bwmm_drop` | 1 | 1 | ON | OFF | **1.0** | **Dropping effect** on BWMM |

**Paper claim**: "BWMM improves single-core MoE throughput by X%."

### Group 2: TP Scaling — Raw Parallelism (nproc=1,2,4)

| Config | TP | EP | BWMM | SP | CF | What it shows |
|--------|----|----|------|----|----|---------------|
| `single_baseline` | 1 | 1 | OFF | OFF | — | (same as Group 1) |
| `tp2_baseline` | **2** | 1 | OFF | OFF | — | **TP scaling**: 1→2 cores |
| `tp4_baseline` | **4** | 1 | OFF | OFF | — | **TP scaling**: 2→4 cores |

**Paper claim**: "TP=4 achieves X% of ideal 4x speedup over TP=1."

### Group 3: TP + BWMM + SP — Combined Optimizations (nproc=2,4)

| Config | TP | EP | BWMM | SP | CF | What it shows |
|--------|----|----|------|----|----|---------------|
| `tp2_bwmm` | 2 | 1 | **ON** | **ON** | — | BWMM+SP at TP=2 (vs `tp2_baseline`) |
| `tp4_bwmm` | 4 | 1 | **ON** | **ON** | — | BWMM+SP at TP=4 (vs `tp4_baseline`) |
| `tp4_bwmm_nosp` | 4 | 1 | ON | **OFF** | — | **SP isolation** (vs `tp4_bwmm`) |
| `tp4_bwmm_drop` | 4 | 1 | ON | ON | **1.0** | **Dropping at scale** (vs `tp4_bwmm`) |

**Paper claims**:
- "BWMM+SP improve TP=4 throughput by X% over vanilla TP."
- "SP adds/reduces Y% overhead at TP=4."
- "Token dropping with CF=1.0 changes throughput by Z%."

### Group 4: Expert Parallelism — EP vs TP (nproc=2,4)

| Config | TP | EP | BWMM | SP | CF | What it shows |
|--------|----|----|------|----|----|---------------|
| `ep2_bwmm` | 1 | **2** | ON | OFF | — | EP=2 (vs `tp2_bwmm` at 2 cores) |
| `ep4_bwmm` | 1 | **4** | ON | OFF | — | EP=4 (vs `tp4_bwmm` at 4 cores) |

**Paper claim**: "At 4 cores, EP outperforms/underperforms TP by X% for uniform inputs, but the gap changes to Y% under skewed load."

### Group 5: Hybrid TP+EP (nproc=4)

| Config | TP | EP | BWMM | SP | CF | What it shows |
|--------|----|----|------|----|----|---------------|
| `tp2_ep2_bwmm` | **2** | **2** | ON | ON | — | Hybrid (vs `tp4_bwmm` and `ep4_bwmm`) |

**Paper claim**: "Hybrid TP=2+EP=2 achieves X% of the throughput of pure TP=4 / EP=4."

---

## How to Run

### Prerequisites

```bash
# Activate the Neuron environment
source setup.sh
```

### Running by nproc Group

Configs must be run with `torchrun --nproc_per_node=N` matching `TP × EP`:

```bash
# === Single core (nproc=1) ===
torchrun --nproc_per_node=1 nxd_benchmark.py --config single_baseline single_bwmm single_bwmm_drop

# === 2 cores (nproc=2) ===
torchrun --nproc_per_node=2 nxd_benchmark.py --config tp2_baseline tp2_bwmm ep2_bwmm

# === 4 cores (nproc=4) ===
torchrun --nproc_per_node=4 nxd_benchmark.py --config tp4_baseline tp4_bwmm tp4_bwmm_nosp tp4_bwmm_drop ep4_bwmm tp2_ep2_bwmm
```

### Running with Neuron Profiling

Add `--profile` to capture hardware-level traces (saved to `profiles/` directory):

```bash
torchrun --nproc_per_node=4 nxd_benchmark.py --profile --config tp4_bwmm
```

### Running a Single Config

```bash
torchrun --nproc_per_node=4 nxd_benchmark.py --config tp4_baseline
```

### Dry Run (Validate Without Executing)

```bash
python3 nxd_benchmark.py --dry-run
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--config NAME [NAME ...]` | Run specific configs (default: all) |
| `--profile` | Enable Neuron profiling → `profiles/` |
| `--dry-run` | Validate configs and inputs without running |
| `--warmup N` | Number of warmup iterations (default: 3) |
| `--runs N` | Number of timed iterations (default: 10) |

---

## Understanding the Output

### Console Output

For each config × input combination, you'll see:
```
Config: tp4_bwmm
  TP=4 + BWMM + SP (vs tp4_baseline → BWMM+SP effect at TP=4)
  TP=4, EP=1, BWMM=ON, cap_factor=None, SP=True

  Input: skewed_b1_s8192.pt
    Warming up (3 iters)...
    Latency: 51.18 +/- 0.23 ms
    Throughput: 160,059 tokens/sec
```

### Summary Table

After all configs run, a summary table is printed:
```
Config               Distribution   SeqLen    Latency (ms)       +-       Tokens/s
single_baseline      uniform          4096           82.15    0.08         49,861
tp4_baseline         uniform          4096           31.99    0.49        128,054
```

### Results File

All results are saved to `benchmark_results.json`:
```json
{
  "config_name": "tp4_baseline",
  "distribution": "uniform",
  "seq_len": 4096,
  "mean_latency_ms": 31.99,
  "std_latency_ms": 0.49,
  "throughput_tokens_per_sec": 128054.0
}
```

### Key Metrics

| Metric | What it measures | How to interpret |
|--------|-----------------|------------------|
| **Latency (ms)** | Wall-clock time for one forward pass | Lower = faster. Compare across configs at same seq_len. |
| **± (ms)** | Standard deviation over 10 runs | High variance suggests cache effects or contention. |
| **Tokens/s** | `seq_len / (latency_ms / 1000)` | Primary throughput metric. Higher = better. |

### Comparing Configs (Example Analysis)

**BWMM Effect** (Group 1):
```
Speedup = single_bwmm.tokens_per_sec / single_baseline.tokens_per_sec
```

**TP Scaling Efficiency** (Group 2):
```
Ideal_speedup = tp_degree
Actual_speedup = tpN_baseline.tokens_per_sec / single_baseline.tokens_per_sec
Efficiency = Actual_speedup / Ideal_speedup × 100%
```

**Skew Sensitivity** (any group):
```
Skew_ratio = config.skewed.tokens_per_sec / config.uniform.tokens_per_sec
# > 1.0 means skewed is faster (fewer experts active = less compute)
# < 1.0 means skewed hurts (load imbalance overhead)
```

---

## Viewing Neuron Profiles

When `--profile` is enabled, the benchmark sets the `NEURON_RT_INSPECT` environment variables at process startup (before any XLA compilation). The **Neuron runtime** then automatically saves hardware-level trace files for the entire process lifetime.

### What Gets Saved

Each profiled run creates a **named, timestamped subdirectory** so you always know which experiment you're looking at:

```
profiles/
└── tp4_baseline_20260317_045500/       # <config(s)>_<YYYYMMDD_HHMMSS>
    ├── run_info.json                   # Metadata: configs, warmup, runs, weights
    ├── i-<instance-id>_pid_<pid1>/     # Worker 0 (NeuronCore 0)
    │   ├── ntrace.pb                   # Neuron execution trace (main data)
    │   ├── trace_info.pb               # Metadata about the trace
    │   ├── cpu_util.pb                 # CPU utilization during the run
    │   └── host_mem.pb                 # Host memory usage
    ├── i-<instance-id>_pid_<pid2>/     # Worker 1
    │   └── ...
    └── ...
```

The `run_info.json` file records exactly which configs, number of warmup/timed runs, and model weights were used. The `ntrace.pb` files contain the hardware-level execution timelines.

### Viewing Profiles

```bash
# View a specific run's profiles
neuron-profile view -d profiles/tp4_baseline_20260317_045500/

# This starts an HTTP server at http://localhost:3001
# If running on a remote EC2 instance, set up SSH port forwarding:
ssh -L 3001:localhost:3001 -L 3002:localhost:3002 user@<instance-ip>
```

### What to Look For

| Observation | Meaning |
|-------------|---------|
| **Low NeuronCore utilization** | Compute is underutilized — likely communication-bound |
| **Large DMA transfer blocks** | Time spent moving data vs computing — BWMM's DMA skip helps here |
| **Long all-reduce / all-to-all** | Communication overhead from TP/EP parallelism |
| **Idle gaps between cores** | Load imbalance between experts (worse with skewed inputs) |
| **Compilation events** | First-run compilation overhead (cached on subsequent runs) |

### Alternative: Perfetto Viewer

You can also export traces and view them in the open-source Perfetto UI (no port forwarding needed):

```bash
# Generate a .pftrace file
neuron-profile view -d profiles/ --output-format pftrace -o profile.pftrace

# Upload profile.pftrace to https://ui.perfetto.dev/
```

### Profiling Overhead

When `--profile` is enabled, expect **higher latency variance** in results. The profiling infrastructure adds overhead to each execution. For clean performance numbers, run without `--profile`. For hardware analysis, run with `--profile` separately.

---

## Known Limitations

1. **Single-core OOM at s8192**: The `single_*` configs may OOM at seq_len=8192 because all 8 experts fit on one NeuronCore with limited HBM. s4096 and s16384 work (different compiler memory strategies). This is not an issue for TP/EP configs.

2. **First-run compilation**: The first invocation of each config triggers Neuron compilation (~2-10 minutes per input shape). Subsequent runs use cached NEFFs and are fast.

3. **Router weight consistency**: The skewed input tensors were generated using the same router weights (`mistral_moe.pt`) loaded into the benchmark. If you change the model weights, regenerate skewed inputs with `python3 generate_inputs.py`.

4. **Profile data is global**: The `NEURON_RT_INSPECT` approach captures traces for the entire process, not per-config. If you run multiple configs in one `torchrun` invocation, all will be captured in one trace. For per-config profiles, run each config separately with `--profile`.

---

## File Layout

```
trn2-moe/
├── nxd_benchmark.py          # Main benchmark harness
├── generate_inputs.py         # Input tensor generator
├── EXPERIMENT_GUIDE.md        # This file
├── setup.sh                   # Environment activation
├── benchmark_results.json     # Output: benchmark metrics
├── weights/
│   ├── uncompiled_model_weights/
│   │   └── mistral_moe.pt     # Mistral 8x7B MoE block weights
│   └── input_weights/
│       ├── uniform_b1_s4096.pt
│       ├── uniform_b1_s8192.pt
│       ├── uniform_b1_s16384.pt
│       ├── skewed_b1_s4096.pt
│       ├── skewed_b1_s8192.pt
│       └── skewed_b1_s16384.pt
└── profiles/                  # Neuron profiler output (when --profile used)
    └── <configs>_<timestamp>/ # e.g. tp4_baseline_20260317_045500/
        ├── run_info.json      # What was profiled (configs, params)
        └── i-<id>_pid_<pid>/  # Per-worker traces
            ├── ntrace.pb      # Hardware execution trace
            ├── trace_info.pb  # Trace metadata
            ├── cpu_util.pb    # CPU utilization
            └── host_mem.pb    # Host memory usage
```

