# NEFF Cache and Profiling Guide

This document describes where NEFF files and profiling data are stored for the Neuron MoE benchmark.

## 1. NEFF Cache Location
The Neuron compiler (`neuron-cc`) caches compiled NEFF files to avoid redundant compilation. 

- **Primary Cache**: `/var/tmp/neuron-compile-cache`
  - This directory contains persistent versions of compiled modules (e.g., `MODULE_...`).
  - Subsequent runs of the benchmark will pull from here if the input shapes and model configurations match.

- **Temporary Compilation Path**: `/tmp/tmp.../graph.neff`
  - During the compilation process, the compiler may write intermediate NEFFs to temporary directories in `/tmp/`.

## 2. Profiling Output
When running with the `--profile` flag in `nxd_benchmark.py`, the Neuron runtime captures hardware-level traces.

- **Profile Directory**: `profiles/` (located in the project root `/home/ubuntu/trn2-moe/`)
- **Structure**: Each run creates a subdirectory named `<config>_<timestamp>/` containing:
  - `ntrace.pb`: Hardware execution trace.
  - `run_info.json`: Metadata about the profiled run.
  - `i-<id>_pid_<pid>/`: Per-worker trace data.

## 3. How to Run the Profiler
To view the profiles, use the `neuron-profile` tool:

```bash
# View a specific run
neuron-profile view -d profiles/<run_name>/

# For remote viewing (SSH port forwarding)
ssh -L 3001:localhost:3001 -L 3002:localhost:3002 user@<instance-ip>
```

Alternatively, you can export to Perfetto:
```bash
neuron-profile view -d profiles/ --output-format pftrace -o profile.pftrace
```
Then upload `profile.pftrace` to [ui.perfetto.dev](https://ui.perfetto.dev/).
