# trn2-moe

CS217 Final Project — Profiling MoE layer kernels on AWS's Trainium2 accelerator.

## Setup

Run the Neuron SDK setup script on a Trainium2 instance:

```bash
bash setup_neuron.sh
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Benchmark

`benchmark.py` is the central entry point for running and profiling NKI kernels.

### Basic usage

Run the default GEMM kernel benchmark on hardware:

```bash
python benchmark.py
```

### Flags

| Flag | Description |
|------|-------------|
| `--kernel {gemm}` | Kernel to benchmark. More kernels (e.g. MoE expert, gating) will be added as the project progresses. Default: `gemm`. |
| `--profile NAME` | Profile the run and save trace files as `NAME.neff` / `NAME.ntff` for viewing in Neuron Profile. |
| `--check` | Verify kernel output against a NumPy reference before benchmarking. |
| `--simulate` | Run the kernel on CPU via the NKI simulator (no Trainium hardware required). |
| `--m M` | GEMM M dimension — rows of A and C (default: 1024). |
| `--k K` | GEMM K dimension — inner / contraction dimension (default: 1024). |
| `--n N` | GEMM N dimension — columns of B and C, must be a multiple of 512 (default: 1024). |
| `--dtype {float16,float32}` | Element data type (default: float32). |
| `--warmup W` | Number of warmup iterations before timing (default: 5). |
| `--iters I` | Number of timed iterations (default: 20). |
| `--seed S` | Random seed for reproducibility (default: 42). |

**Reserved flags for future MoE kernels:**

| Flag | Description |
|------|-------------|
| `--num-experts` | Number of experts (default: 8). |
| `--top-k` | Top-k experts routed per token (default: 2). |
| `--hidden-size` | Hidden dimension (default: 4096). |
| `--intermediate-size` | Expert FFN intermediate dimension (default: 14336). |

### Examples

Profile a 4096×4096 GEMM and generate trace files:

```bash
python benchmark.py --kernel gemm --m 4096 --k 4096 --n 4096 --profile gemm_4k
```

Check correctness using the CPU simulator (no hardware needed):

```bash
python benchmark.py --check --simulate
```

Benchmark in float16 with custom dimensions:

```bash
python benchmark.py --dtype float16 --m 2048 --k 2048 --n 2048
```

### Viewing profiles

After generating a profile, open the trace in Neuron Profile:

```bash
neuron-profile view -n <name>.neff -s <name>.ntff
```

## Project structure

| File | Purpose |
|------|---------|
| `pytorch_moe.py` | Reference PyTorch MoE block (Mixtral-style). |
| `gemm_nki.py` | Tiled GEMM kernel written in NKI for Trainium. |
| `benchmark.py` | CLI benchmark and profiling harness. |
| `cs149/` | Reference NKI kernels and test harness from CS149. |
| `setup_neuron.sh` | Neuron SDK installation script. |
| `requirements.txt` | Python dependencies. |
