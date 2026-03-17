#!/bin/bash
# Full MoE experiment sweep — runs each config individually
# Results accumulate in benchmark_results.json (merge, not overwrite)
set -e

source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate

# Clear previous results
rm -f benchmark_results.json

echo "========================================"
echo "MoE EXPERIMENT SWEEP"
echo "Started: $(date)"
echo "========================================"

# Group 1: Single core (nproc=1)
for config in single_baseline single_bwmm single_bwmm_drop; do
    echo ""
    echo ">>> [$config] nproc=1 — $(date)"
    torchrun --nproc_per_node=1 nxd_benchmark.py --config $config 2>&1 | \
        grep -E "Expert load|Drop rate|Latency|Throughput|✓|✗|SUMMARY|Config |Results saved|ERROR|MoE layer"
    echo "<<< [$config] done — $(date)"
done

# Group 2: Two cores (nproc=2)
for config in tp2_baseline tp2_bwmm ep2_bwmm; do
    echo ""
    echo ">>> [$config] nproc=2 — $(date)"
    torchrun --nproc_per_node=2 nxd_benchmark.py --config $config 2>&1 | \
        grep -E "Expert load|Drop rate|Latency|Throughput|✓|✗|SUMMARY|Config |Results saved|ERROR|MoE layer"
    echo "<<< [$config] done — $(date)"
done

# Group 3: Four cores (nproc=4)
for config in tp4_baseline tp4_bwmm tp4_bwmm_nosp tp4_bwmm_drop ep4_bwmm tp2_ep2_bwmm; do
    echo ""
    echo ">>> [$config] nproc=4 — $(date)"
    torchrun --nproc_per_node=4 nxd_benchmark.py --config $config 2>&1 | \
        grep -E "Expert load|Drop rate|Latency|Throughput|✓|✗|SUMMARY|Config |Results saved|ERROR|MoE layer"
    echo "<<< [$config] done — $(date)"
done

echo ""
echo "========================================"
echo "SWEEP COMPLETE — $(date)"
echo "Results: benchmark_results.json"
echo "========================================"
