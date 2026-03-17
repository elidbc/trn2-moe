"""
NxD MoE Benchmark Harness

Initializes multiple NxD MoE layer configurations with Mistral 8x7B dimensions
and provides profiling infrastructure for comparing optimizations.

Configurations tested:
  1. baseline_tp       — TP=8, no BWMM (uses standard matmul path)
  2. bwmm_dropless     — TP=8, BWMM with DMA skip, dropless
  3. bwmm_dropping     — TP=8, BWMM with dropping (capacity_factor=1.0)
  4. ep_only           — EP=8, BWMM dropless
  5. tp_ep_hybrid      — TP=2 + EP=4, BWMM dropless

Usage:
    # Dry run (verify configs without Neuron):
    python3 nxd_benchmark.py --dry-run

    # Full run on Neuron (requires torchrun for TP/EP):
    torchrun --nproc_per_node=8 nxd_benchmark.py

    # Run specific config:
    python3 nxd_benchmark.py --dry-run --config bwmm_dropless

    # With Neuron profiling:
    python3 nxd_benchmark.py --profile

    # Profile specific config + input:
    python3 nxd_benchmark.py --profile --config bwmm_dropless
"""

import argparse
import os
import sys
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch


def rprint(*args, **kwargs):
    """Print only from rank 0 (or when not in distributed mode)."""
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

# ============================================================================
# Mistral 8x7B MoE dimensions
# ============================================================================
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 14336
NUM_EXPERTS = 8
TOP_K = 2
HIDDEN_ACT = "silu"
BLOCK_SIZE = 512  # BWMM block size for Trn2
LNC_CONFIG = 2    # LNC2 for Trn2

INPUT_DIR = "weights/input_weights"

# Same model weights used to generate skewed inputs — this ensures the skewed
# distribution actually routes to the intended experts when run through the
# NxD MoE layer's router.
MODEL_WEIGHTS_PATH = "weights/uncompiled_model_weights/mistral_moe.pt"

# Neuron profiler output directory
PROFILE_DIR = "profiles"

# ============================================================================
# Configuration dataclasses
# ============================================================================

@dataclass
class MoEConfig:
    """Configuration for a single NxD MoE experiment."""
    name: str
    description: str
    
    # Parallelism
    tp_degree: int = 8
    ep_degree: int = 1
    
    # Router
    sequence_parallel_enabled: bool = False
    
    # Expert MLPs
    capacity_factor: Optional[float] = None  # None = dropless
    normalize_top_k_affinities: bool = True
    
    # BWMM
    use_bwmm: bool = True
    block_size: int = BLOCK_SIZE
    logical_nc_config: int = LNC_CONFIG
    skip_dma_token: bool = True
    skip_dma_weight: bool = True
    use_shard_on_block_dynamic_while: bool = False
    use_shard_on_intermediate_dynamic_while: bool = False


# ============================================================================
# Define experiment configurations
# ============================================================================

CONFIGS = {
    # =================================================================
    # Group 1: Single NeuronCore (nproc=1)
    #   Isolates BWMM effect and token dropping on a single core.
    # =================================================================
    "single_baseline": MoEConfig(
        name="single_baseline",
        description="Single core: no parallelism, no BWMM",
        tp_degree=1,
        ep_degree=1,
        sequence_parallel_enabled=False,
        use_bwmm=False,
        skip_dma_token=False,
        skip_dma_weight=False,
    ),
    
    "single_bwmm": MoEConfig(
        name="single_bwmm",
        description="Single core: BWMM enabled (vs single_baseline → BWMM effect)",
        tp_degree=1,
        ep_degree=1,
        sequence_parallel_enabled=False,
        use_bwmm=True,
    ),
    
    "single_bwmm_drop": MoEConfig(
        name="single_bwmm_drop",
        description="Single core: BWMM + token dropping CF=1.0 (vs single_bwmm → dropping effect)",
        tp_degree=1,
        ep_degree=1,
        sequence_parallel_enabled=False,
        capacity_factor=1.0,
        use_bwmm=True,
    ),
    
    # =================================================================
    # Group 2: TP Scaling without BWMM (nproc=1,2,4)
    #   Shows raw tensor parallelism scaling. single_baseline is TP=1.
    # =================================================================
    "tp2_baseline": MoEConfig(
        name="tp2_baseline",
        description="TP=2: tensor parallelism only, no BWMM (vs single_baseline → TP effect)",
        tp_degree=2,
        ep_degree=1,
        sequence_parallel_enabled=False,
        use_bwmm=False,
        skip_dma_token=False,
        skip_dma_weight=False,
    ),
    
    "tp4_baseline": MoEConfig(
        name="tp4_baseline",
        description="TP=4: tensor parallelism only, no BWMM (vs tp2_baseline → TP scaling)",
        tp_degree=4,
        ep_degree=1,
        sequence_parallel_enabled=False,
        use_bwmm=False,
        skip_dma_token=False,
        skip_dma_weight=False,
    ),
    
    # =================================================================
    # Group 3: TP + BWMM + SP (nproc=2,4)
    #   Compares BWMM+SP benefit at each TP degree, plus token dropping.
    # =================================================================
    "tp2_bwmm": MoEConfig(
        name="tp2_bwmm",
        description="TP=2 + BWMM + SP (vs tp2_baseline → BWMM+SP effect at TP=2)",
        tp_degree=2,
        ep_degree=1,
        sequence_parallel_enabled=True,
        use_bwmm=True,
    ),
    
    "tp4_bwmm": MoEConfig(
        name="tp4_bwmm",
        description="TP=4 + BWMM + SP (vs tp4_baseline → BWMM+SP effect at TP=4)",
        tp_degree=4,
        ep_degree=1,
        sequence_parallel_enabled=True,
        use_bwmm=True,
    ),
    
    "tp4_bwmm_nosp": MoEConfig(
        name="tp4_bwmm_nosp",
        description="TP=4 + BWMM, SP off (vs tp4_bwmm → SP overhead/benefit isolation)",
        tp_degree=4,
        ep_degree=1,
        sequence_parallel_enabled=False,
        use_bwmm=True,
    ),
    
    "tp4_bwmm_drop": MoEConfig(
        name="tp4_bwmm_drop",
        description="TP=4 + BWMM + SP + dropping CF=1.0 (vs tp4_bwmm → dropping at scale)",
        tp_degree=4,
        ep_degree=1,
        sequence_parallel_enabled=True,
        capacity_factor=1.0,
        use_bwmm=True,
    ),
    
    # =================================================================
    # Group 4: Expert Parallelism (nproc=2,4)
    #   Pure EP — compare vs same-core-count TP configs.
    # =================================================================
    "ep2_bwmm": MoEConfig(
        name="ep2_bwmm",
        description="EP=2 + BWMM (vs tp2_bwmm → EP vs TP at 2 cores)",
        tp_degree=1,
        ep_degree=2,
        sequence_parallel_enabled=False,
        use_bwmm=True,
    ),
    
    "ep4_bwmm": MoEConfig(
        name="ep4_bwmm",
        description="EP=4 + BWMM (vs tp4_bwmm → EP vs TP at 4 cores)",
        tp_degree=1,
        ep_degree=4,
        sequence_parallel_enabled=False,
        use_bwmm=True,
    ),
    
    # =================================================================
    # Group 5: Hybrid TP+EP (nproc=4)
    #   Combined parallelism — compare vs pure TP and pure EP.
    # =================================================================
    "tp2_ep2_bwmm": MoEConfig(
        name="tp2_ep2_bwmm",
        description="TP=2 + EP=2 + BWMM + SP (vs tp4_bwmm/ep4_bwmm → hybrid effect)",
        tp_degree=2,
        ep_degree=2,
        sequence_parallel_enabled=True,
        use_bwmm=True,
    ),
}


# ============================================================================
# Model weight loading
# ============================================================================

def load_model_weights(moe_layer, model_path: str = MODEL_WEIGHTS_PATH):
    """
    Load saved Mistral MoE weights into an NxD MoE layer.
    
    Maps the router weights from our saved MixtralSparseMoeBlock format
    into the NxD RouterTopK format. This ensures the NxD layer uses the
    same router as was used to generate the skewed input tensors.
    
    Args:
        moe_layer: The NxD MoE layer to load weights into
        model_path: Path to the saved .pt state dict
    
    Returns:
        True if weights were loaded successfully, False otherwise
    """
    if not os.path.exists(model_path):
        rprint(f"  WARNING: Model weights not found at {model_path}")
        rprint(f"  Using randomly initialized weights (skewed inputs may not be skewed!)")
        return False
    
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    
    # Load router weights
    # Our format: router.router.weight -> [num_experts, hidden_size]
    # NxD RouterTopK also uses a linear layer with the same shape
    if "router.router.weight" in state_dict:
        router_weight = state_dict["router.router.weight"]
        loaded = False
        # Try NxD router attribute paths (NxD RouterTopK uses linear_router.weight)
        for attr_path in ["router.linear_router.weight", "router.linear.weight", "router.weight"]:
            try:
                obj = moe_layer
                parts = attr_path.split(".")
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                param = getattr(obj, parts[-1])
                param.data.copy_(router_weight)
                rprint(f"  ✓ Loaded router weights via {attr_path}: {router_weight.shape}")
                loaded = True
                break
            except (AttributeError, RuntimeError):
                continue
        if not loaded:
            rprint(f"  ⚠ Could not auto-load router weights into NxD layer")
            rprint(f"    Available router params: {[n for n, _ in moe_layer.router.named_parameters()]}")
    
    # Load expert weights (w1, w2, w3)
    expert_keys = {k: v for k, v in state_dict.items() if k.startswith("w")}
    if expert_keys:
        rprint(f"  ℹ Expert weights available: {list(expert_keys.keys())}")
        rprint(f"    Shapes: { {k: list(v.shape) for k, v in expert_keys.items()} }")
        # Note: Direct loading into NxD ExpertMLPsV2 depends on the specific
        # parallelism config. For benchmarking, the key thing is the router
        # weights match — expert weights affect output values but not the
        # routing pattern or computational profile we're measuring.
    
    return True


# ============================================================================
# NxD MoE Layer Builder
# ============================================================================

def build_nxd_moe_layer(config: MoEConfig):
    """
    Build an NxD MoE layer from a config.
    
    Returns the MoE layer and a dict of component objects for inspection.
    Requires neuronx_distributed to be installed.
    """
    from neuronx_distributed.modules.moe import MoE, routing
    from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
    from neuronx_distributed.modules.moe.moe_configs import (
        RoutedExpertsMLPOpsConfig,
        BlockwiseMatmulConfig,
    )
    
    # --- Router ---
    router = routing.RouterTopK(
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN_SIZE,
        sequence_parallel_enabled=config.sequence_parallel_enabled,
    )
    
    # --- Expert MLP config ---
    routed_experts_config = RoutedExpertsMLPOpsConfig(
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_act=HIDDEN_ACT,
        glu_mlp=True,
        capacity_factor=config.capacity_factor,
        normalize_top_k_affinities=config.normalize_top_k_affinities,
        # EP requires enable_spmd_rank so ExpertMLPsV2 can determine local experts
        enable_spmd_rank=(config.ep_degree > 1),
    )
    
    # --- BWMM config ---
    # ExpertMLPsV2 always reads blockwise_matmul_config attributes, so use default() not None
    blockwise_config = BlockwiseMatmulConfig.default()
    if config.use_bwmm:
        bwmm_kwargs = {
            "block_size": config.block_size,
            "logical_nc_config": config.logical_nc_config,
            "skip_dma_token": config.skip_dma_token,
            "skip_dma_weight": config.skip_dma_weight,
        }
        if config.use_shard_on_block_dynamic_while:
            bwmm_kwargs["use_shard_on_block_dynamic_while"] = True
        if config.use_shard_on_intermediate_dynamic_while:
            bwmm_kwargs["use_shard_on_intermediate_dynamic_while"] = True
        
        blockwise_config = BlockwiseMatmulConfig.from_kwargs(**bwmm_kwargs)
    
    # --- Expert MLPs ---
    expert_mlps = ExpertMLPsV2(
        routed_experts_mlp_config=routed_experts_config,
        blockwise_matmul_config=blockwise_config,
        sequence_parallel_enabled=config.sequence_parallel_enabled,
    )
    
    # --- Full MoE layer ---
    # sequence_dimension=0 matches RouterTopK's default when SP is enabled.
    # Input layout: [S, B, H] for training (sequence dim=0).
    moe_layer = MoE(
        router=router,
        expert_mlps=expert_mlps,
        sequence_parallel_enabled=config.sequence_parallel_enabled,
        sequence_dimension=0 if config.sequence_parallel_enabled else None,
    )
    
    return moe_layer, {
        "router": router,
        "expert_mlps": expert_mlps,
        "blockwise_config": blockwise_config,
        "routed_experts_config": routed_experts_config,
    }


def ensure_distributed_init():
    """Initialize torch.distributed with XLA backend for Neuron, gloo fallback for CPU.
    
    When launched via torchrun, torch.distributed is NOT yet initialized — torchrun
    only sets env vars (MASTER_ADDR, RANK, WORLD_SIZE, etc.). We must call
    init_process_group ourselves, preferring XLA for Neuron hardware.
    """
    import torch.distributed as dist
    if dist.is_initialized():
        rprint(f"  torch.distributed already initialized: "
              f"world_size={dist.get_world_size()}, rank={dist.get_rank()}")
        return dist.get_world_size()
    
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    
    # Try XLA backend first (required for Neuron hardware forward passes)
    try:
        import torch_xla.distributed.xla_backend
        dist.init_process_group(backend="xla")
        rprint(f"  Initialized torch.distributed: backend=xla, "
              f"world_size={dist.get_world_size()}, rank={dist.get_rank()}")
    except Exception as e:
        rprint(f"  XLA backend not available ({e}), falling back to gloo")
        dist.init_process_group(backend="gloo")
        rprint(f"  Initialized torch.distributed: backend=gloo, "
              f"world_size={dist.get_world_size()}, rank={dist.get_rank()}")
        rprint(f"  ⚠ gloo backend: layer init works but forward passes require XLA (Neuron)")
    
    return dist.get_world_size()


def setup_parallel_state(config: MoEConfig):
    """Set up NxD parallel state for a config, destroying any previous state first.
    
    Returns True if setup succeeded, False if config is incompatible with
    current world_size (e.g. TP=8 with world_size=1).
    """
    import torch.distributed as dist
    from neuronx_distributed.parallel_layers import parallel_state
    
    world_size = dist.get_world_size()
    required = config.tp_degree * config.ep_degree
    
    if required > world_size:
        rprint(f"  ⚠ Skipping: config requires TP*EP={required} workers "
              f"but world_size={world_size}")
        rprint(f"    Run with: torchrun --nproc_per_node={required} nxd_benchmark.py")
        return False
    
    # Destroy previous state if it exists
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=config.tp_degree,
        expert_model_parallel_size=config.ep_degree,
    )
    rprint(f"  ✓ Parallel state: TP={config.tp_degree}, EP={config.ep_degree}")
    return True


# ============================================================================
# Neuron Profiler
# ============================================================================

class NeuronProfiler:
    """
    Profiling via NEURON_RT_INSPECT environment variables.
    
    Set at process startup in main() before any XLA compilation.
    The Neuron runtime saves trace files to the output directory.
    View later: neuron-profile view -d profiles/
    """
    
    def __init__(self, output_dir: str = PROFILE_DIR):
        self.output_dir = output_dir
        self.enabled = True
        os.makedirs(output_dir, exist_ok=True)
        rprint(f"  ✓ Profiling enabled → {output_dir}/")
        rprint(f"    View later: neuron-profile view -d {output_dir}/")
    
    def profile(self, run_name: str):
        """No-op context manager — profiling is global via env vars."""
        return _NullContext()


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ============================================================================
# Benchmark Runner
# ============================================================================

@dataclass 
class BenchmarkResult:
    """Result from a single benchmark run."""
    config_name: str
    input_name: str
    seq_len: int
    distribution: str  # "uniform" or "skewed"
    num_warmup: int
    num_runs: int
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    per_run_latencies_ms: Optional[list] = None  # all individual run times
    throughput_tokens_per_sec: float = 0.0
    # Expert load: tokens routed to each expert (list of counts, length=num_experts)
    expert_load: Optional[list] = None
    # Token drop rate: fraction of token-expert assignments dropped (CF configs)
    token_drop_rate: Optional[float] = None
    tokens_dropped: Optional[int] = None
    tokens_total: Optional[int] = None  # total token-expert assignments (T * top_k)
    profile_dir: Optional[str] = None
    error: Optional[str] = None


class BenchmarkRunner:
    """Runs benchmarks across configurations and input tensors."""
    
    def __init__(self, num_warmup: int = 3, num_runs: int = 10, enable_profiling: bool = False):
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        self.enable_profiling = enable_profiling
        self.profiler = NeuronProfiler() if enable_profiling else None
        self.results: list[BenchmarkResult] = []
    
    def load_inputs(self) -> dict[str, torch.Tensor]:
        """Load all input tensors from the input directory."""
        inputs = {}
        if not os.path.isdir(INPUT_DIR):
            rprint(f"  WARNING: Input directory '{INPUT_DIR}' not found.")
            rprint(f"  Run 'python3 generate_inputs.py' first to create input tensors.")
            return inputs
        
        for fname in sorted(os.listdir(INPUT_DIR)):
            if fname.endswith(".pt") and (fname.startswith("uniform_") or fname.startswith("skewed_")):
                path = os.path.join(INPUT_DIR, fname)
                tensor = torch.load(path, map_location="cpu", weights_only=True)
                inputs[fname] = tensor
                rprint(f"  Loaded: {fname} | shape={tensor.shape} | dtype={tensor.dtype}")
        
        return inputs
    
    def _timed_run(self, moe_layer, x, num_iters):
        """Run model num_iters times and return list of latencies in ms."""
        latencies = []
        with torch.no_grad():
            for _ in range(num_iters):
                start = time.perf_counter()
                _ = moe_layer(x)
                # Sync XLA device for accurate timing
                try:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()
                except ImportError:
                    pass
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
        return latencies
    
    def run_single(
        self,
        moe_layer: torch.nn.Module,
        config: MoEConfig,
        input_tensor: torch.Tensor,
        input_name: str,
    ) -> BenchmarkResult:
        """Run a benchmark for a single config x input combination."""
        parts = input_name.replace(".pt", "").split("_")
        distribution = parts[0]
        seq_len = int(parts[-1].replace("s", ""))
        
        profile_run_name = f"{config.name}_{distribution}_s{seq_len}"
        profile_dir = os.path.join(PROFILE_DIR, profile_run_name) if self.enable_profiling else None
        
        result = BenchmarkResult(
            config_name=config.name,
            input_name=input_name,
            seq_len=seq_len,
            distribution=distribution,
            num_warmup=self.num_warmup,
            num_runs=self.num_runs,
            profile_dir=profile_dir,
        )
        
        try:
            x = input_tensor.to(dtype=torch.bfloat16)
            
            # Move to XLA device if available
            try:
                import torch_xla.core.xla_model as xm
                device = xm.xla_device()
                x = x.to(device)
            except (ImportError, RuntimeError):
                pass
            
            # For SP-enabled configs, input is [S, B, H] (sequence_dimension=0)
            # Our saved tensors are [B, S, H] so transpose if needed
            if hasattr(config, 'sequence_parallel_enabled') and config.sequence_parallel_enabled:
                # [B, S, H] -> [S, B, H]
                x = x.transpose(0, 1)
            
            # --- Capture expert routing info (outside timed runs) ---
            try:
                with torch.no_grad():
                    router = moe_layer.router
                    _, _, expert_index = router(x)
                    # Sync for XLA
                    try:
                        import torch_xla.core.xla_model as xm
                        xm.mark_step()
                    except ImportError:
                        pass
                    # expert_index: (T, top_k) — which experts each token was routed to
                    expert_index_cpu = expert_index.cpu()
                    total_assignments = expert_index_cpu.numel()  # T * top_k
                    
                    # Expert load: count tokens routed to each expert
                    load_counts = torch.zeros(NUM_EXPERTS, dtype=torch.long)
                    for e in range(NUM_EXPERTS):
                        load_counts[e] = (expert_index_cpu == e).sum().item()
                    result.expert_load = load_counts.tolist()
                    result.tokens_total = total_assignments
                    
                    # Token drop rate (only meaningful with capacity_factor)
                    if config.capacity_factor is not None:
                        capacity_per_expert = int(
                            (total_assignments * config.capacity_factor) / NUM_EXPERTS
                        )
                        # Tokens exceeding capacity per expert are dropped
                        dropped = sum(
                            max(0, count - capacity_per_expert) 
                            for count in load_counts.tolist()
                        )
                        result.tokens_dropped = dropped
                        result.token_drop_rate = dropped / total_assignments if total_assignments > 0 else 0.0
                    else:
                        result.tokens_dropped = 0
                        result.token_drop_rate = 0.0
                    
                    rprint(f"    Expert load: {result.expert_load}")
                    if config.capacity_factor is not None:
                        rprint(f"    Drop rate: {result.token_drop_rate:.1%} ({result.tokens_dropped}/{result.tokens_total})")
            except Exception as e:
                rprint(f"    ⚠ Could not capture routing info: {e}")
            
            # Warmup (outside profiler)
            rprint(f"    Warming up ({self.num_warmup} iters)...")
            self._timed_run(moe_layer, x, self.num_warmup)
            
            # Timed runs with optional profiling
            if self.enable_profiling and self.profiler:
                rprint(f"    Profiling enabled -> {profile_dir}")
                with self.profiler.profile(profile_run_name):
                    latencies = self._timed_run(moe_layer, x, self.num_runs)
            else:
                latencies = self._timed_run(moe_layer, x, self.num_runs)
            
            lt = torch.tensor(latencies)
            result.mean_latency_ms = lt.mean().item()
            result.std_latency_ms = lt.std().item()
            result.min_latency_ms = lt.min().item()
            result.max_latency_ms = lt.max().item()
            result.p50_latency_ms = lt.median().item()
            result.p95_latency_ms = lt.quantile(0.95).item()
            result.p99_latency_ms = lt.quantile(0.99).item()
            result.per_run_latencies_ms = [round(l, 3) for l in latencies]
            result.throughput_tokens_per_sec = seq_len / (result.mean_latency_ms / 1000)
            
        except Exception as e:
            result.error = str(e)
        
        return result
    
    def run_all(self, configs: dict[str, MoEConfig], inputs: dict[str, torch.Tensor]):
        """Run benchmarks for all config x input combinations."""
        for config_name, config in configs.items():
            rprint(f"\n{'='*70}")
            rprint(f"Config: {config.name}")
            rprint(f"  {config.description}")
            rprint(f"  TP={config.tp_degree}, EP={config.ep_degree}, "
                  f"BWMM={'ON' if config.use_bwmm else 'OFF'}, "
                  f"cap_factor={config.capacity_factor}, SP={config.sequence_parallel_enabled}")
            rprint(f"{'='*70}")
            
            # Set up parallel state for this config
            if not setup_parallel_state(config):
                for input_name in inputs:
                    parts = input_name.replace(".pt", "").split("_")
                    distribution = parts[0]
                    seq_len = int(parts[-1].replace("s", ""))
                    self.results.append(BenchmarkResult(
                        config_name=config_name,
                        input_name=input_name,
                        seq_len=seq_len,
                        distribution=distribution,
                        num_warmup=self.num_warmup,
                        num_runs=self.num_runs,
                        error=f"Requires TP*EP={config.tp_degree*config.ep_degree} workers",
                    ))
                continue
            
            try:
                moe_layer, components = build_nxd_moe_layer(config)
                
                # Load the same model weights used to generate skewed inputs
                rprint(f"  Loading model weights from {MODEL_WEIGHTS_PATH}...")
                load_model_weights(moe_layer, MODEL_WEIGHTS_PATH)
                
                moe_layer = moe_layer.to(dtype=torch.bfloat16)
                
                # Move model to XLA device for Neuron execution
                try:
                    import torch_xla.core.xla_model as xm
                    device = xm.xla_device()
                    moe_layer = moe_layer.to(device)
                    rprint(f"  ✓ Model moved to XLA device")
                except (ImportError, RuntimeError) as e:
                    rprint(f"  ⚠ Could not move to XLA: {e} (running on CPU)")
                    device = torch.device("cpu")
                
                moe_layer.eval()
                rprint(f"  ✓ MoE layer initialized successfully")
            except Exception as e:
                rprint(f"  ✗ Failed to initialize MoE layer: {e}")
                for input_name in inputs:
                    parts = input_name.replace(".pt", "").split("_")
                    distribution = parts[0]
                    seq_len = int(parts[-1].replace("s", ""))
                    self.results.append(BenchmarkResult(
                        config_name=config_name,
                        input_name=input_name,
                        seq_len=seq_len,
                        distribution=distribution,
                        num_warmup=self.num_warmup,
                        num_runs=self.num_runs,
                        error=str(e),
                    ))
                continue
            
            for input_name, input_tensor in inputs.items():
                rprint(f"\n  Input: {input_name}")
                result = self.run_single(moe_layer, config, input_tensor, input_name)
                self.results.append(result)
                
                if result.error:
                    rprint(f"    ✗ Error: {result.error}")
                else:
                    rprint(f"    Latency: {result.mean_latency_ms:.2f} +/- {result.std_latency_ms:.2f} ms")
                    rprint(f"    Throughput: {result.throughput_tokens_per_sec:,.0f} tokens/sec")
                    if result.profile_dir:
                        rprint(f"    Profile:  neuron-profile view -d {result.profile_dir}")
    
    def print_summary_table(self):
        """Print a summary table of all benchmark results."""
        rprint(f"\n{'='*110}")
        rprint("BENCHMARK SUMMARY")
        rprint(f"{'='*110}")
        
        header = (f"{'Config':<20} {'Distribution':<12} {'SeqLen':>8} "
                  f"{'Latency (ms)':>15} {'+-':>8} {'Tokens/s':>14} {'Profile':>20} {'Status':<10}")
        rprint(header)
        rprint("-" * 110)
        
        for r in self.results:
            prof = os.path.basename(r.profile_dir) if r.profile_dir else "-"
            if r.error:
                rprint(f"{r.config_name:<20} {r.distribution:<12} {r.seq_len:>8} "
                      f"{'-':>15} {'-':>8} {'-':>14} {prof:>20} {'ERROR':<10}")
            else:
                rprint(f"{r.config_name:<20} {r.distribution:<12} {r.seq_len:>8} "
                      f"{r.mean_latency_ms:>15.2f} {r.std_latency_ms:>7.2f} "
                      f"{r.throughput_tokens_per_sec:>14,.0f} {prof:>20} {'OK':<10}")
        
        rprint(f"{'='*110}")
        
        if any(r.profile_dir for r in self.results if not r.error):
            rprint(f"\nProfile data saved to '{PROFILE_DIR}/'")
            rprint(f"View profiles with: neuron-profile view -d {PROFILE_DIR}/<run_name>")
    
    def save_results(self, path: str = "benchmark_results.json"):
        """Save results to JSON, merging with any existing results."""
        new_results = [asdict(r) for r in self.results]
        
        # Load existing results and merge
        existing = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, ValueError):
                existing = []
        
        # Build lookup for dedup: replace existing entries with same key
        def result_key(r):
            return (r["config_name"], r["distribution"], r["seq_len"])
        
        merged = {result_key(r): r for r in existing}
        for r in new_results:
            merged[result_key(r)] = r  # new results overwrite old ones
        
        all_results = list(merged.values())
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2)
        rprint(f"\nResults saved to {path} ({len(all_results)} total, {len(new_results)} new)")


# ============================================================================
# Dry run mode
# ============================================================================

def dry_run(config_names: Optional[list[str]] = None):
    """Verify configuration structure and input loading."""
    rprint("=" * 70)
    rprint("DRY RUN MODE - Validating configs and inputs")
    rprint("=" * 70)
    
    configs_to_test = CONFIGS
    if config_names:
        configs_to_test = {k: v for k, v in CONFIGS.items() if k in config_names}
    
    # Config table
    rprint(f"\n{'Config':<20} {'TP':>4} {'EP':>4} {'BWMM':>6} {'Cap.F':>8} {'SP':>4} {'DMA Skip':>10}")
    rprint("-" * 70)
    for name, cfg in configs_to_test.items():
        cap_f = str(cfg.capacity_factor) if cfg.capacity_factor is not None else "None"
        dma = f"T={cfg.skip_dma_token},W={cfg.skip_dma_weight}"
        rprint(f"{name:<20} {cfg.tp_degree:>4} {cfg.ep_degree:>4} {'ON' if cfg.use_bwmm else 'OFF':>6} {cap_f:>8} {'ON' if cfg.sequence_parallel_enabled else 'OFF':>4} {dma:>10}")
    
    # Model weights
    rprint(f"\nModel weights: {MODEL_WEIGHTS_PATH}")
    if os.path.exists(MODEL_WEIGHTS_PATH):
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        rprint(f"  Found ({len(state_dict)} keys)")
        for key, val in state_dict.items():
            rprint(f"    {key}: {list(val.shape)} ({val.dtype})")
    else:
        rprint(f"  Not found!")
    
    # Inputs
    rprint(f"\nInput tensors from '{INPUT_DIR}':")
    runner = BenchmarkRunner()
    inputs = runner.load_inputs()
    if not inputs:
        rprint("  No input tensors found. Run 'python3 generate_inputs.py' first.")
    
    # NxD availability
    rprint("\nChecking NxD availability...")
    try:
        from neuronx_distributed.modules.moe import MoE, routing
        from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
        from neuronx_distributed.modules.moe.moe_configs import (
            RoutedExpertsMLPOpsConfig,
            BlockwiseMatmulConfig,
        )
        rprint("  neuronx_distributed is available")
        
        # Initialize distributed for layer construction
        world_size = ensure_distributed_init()
        
        for name, cfg in configs_to_test.items():
            required = cfg.tp_degree * cfg.ep_degree
            if required > world_size:
                rprint(f"  ⚠ Config '{name}': requires {required} workers (have {world_size}), "
                      f"use torchrun --nproc_per_node={required}")
                continue
            try:
                if not setup_parallel_state(cfg):
                    continue
                moe_layer, components = build_nxd_moe_layer(cfg)
                load_model_weights(moe_layer, MODEL_WEIGHTS_PATH)
                rprint(f"  ✓ Config '{name}' initialized + weights loaded")
            except Exception as e:
                rprint(f"  ✗ Config '{name}' failed: {e}")
                
    except ImportError as e:
        rprint(f"  neuronx_distributed not available: {e}")
        rprint("  Layer initialization will be tested on Neuron hardware.")
    
    # Neuron profiler
    rprint("\nChecking Neuron profiler availability...")
    try:
        import torch_neuronx
        rprint(f"  torch_neuronx available")
        try:
            from torch_neuronx.experimental import profiler
            rprint(f"  Experimental profiler API available")
        except ImportError:
            rprint(f"  Experimental profiler not found, will use NEURON_PROFILE env var")
            rprint(f"    Run with: NEURON_PROFILE={PROFILE_DIR} python3 nxd_benchmark.py")
    except ImportError:
        rprint(f"  torch_neuronx not available")
    
    rprint("\nDry run complete.")


# ============================================================================
# Main
# ============================================================================

def main():
    global MODEL_WEIGHTS_PATH, PROFILE_DIR
    
    parser = argparse.ArgumentParser(description="NxD MoE Benchmark Harness")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configs without running benchmarks")
    parser.add_argument("--config", type=str, default=None,
                        help="Run specific config (comma-separated names)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of timed iterations")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON path")
    parser.add_argument("--profile", action="store_true",
                        help="Enable Neuron profiling (saves traces to profiles/)")
    parser.add_argument("--profile-dir", type=str, default=PROFILE_DIR,
                        help=f"Directory for profile output (default: {PROFILE_DIR})")
    parser.add_argument("--weights", type=str, default=MODEL_WEIGHTS_PATH,
                        help=f"Path to model weights (default: {MODEL_WEIGHTS_PATH})")
    args = parser.parse_args()
    
    MODEL_WEIGHTS_PATH = args.weights
    PROFILE_DIR = args.profile_dir
    
    config_names = args.config.split(",") if args.config else None
    
    if args.dry_run:
        dry_run(config_names)
        return
    
    # Full benchmark run
    rprint("=" * 70)
    rprint("NxD MoE BENCHMARK")
    rprint(f"Mistral 8x7B dims: hidden={HIDDEN_SIZE}, intermediate={INTERMEDIATE_SIZE}")
    rprint(f"Experts: {NUM_EXPERTS}, top_k={TOP_K}")
    rprint(f"Warmup: {args.warmup}, Runs: {args.runs}")
    rprint(f"Profiling: {'ENABLED -> ' + PROFILE_DIR if args.profile else 'disabled'}")
    rprint(f"Model weights: {MODEL_WEIGHTS_PATH}")
    rprint("=" * 70)
    
    # Initialize torch.distributed (auto-detects single vs multi-process)
    rprint("\nInitializing distributed environment...")
    
    # Set profiling env vars BEFORE any XLA/NEFF operations so the Neuron
    # runtime captures traces for the entire process lifetime.
    # Directory name includes config names + timestamp for identification.
    if args.profile:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_label = "_".join(config_names) if config_names else "all"
        run_profile_dir = os.path.join(
            os.path.abspath(PROFILE_DIR),
            f"{config_label}_{timestamp}"
        )
        os.makedirs(run_profile_dir, exist_ok=True)
        os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
        os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = run_profile_dir
        rprint(f"\nNEURON_RT_INSPECT_OUTPUT_DIR={run_profile_dir}")
        
        # Write metadata so you know exactly what this profile captured
        import json as _json
        run_info = {
            "timestamp": timestamp,
            "configs": config_names or list(CONFIGS.keys()),
            "warmup": args.warmup,
            "runs": args.runs,
            "weights": MODEL_WEIGHTS_PATH,
        }
        with open(os.path.join(run_profile_dir, "run_info.json"), "w") as f:
            _json.dump(run_info, f, indent=2)
        rprint(f"  Saved run_info.json with config metadata")
    
    world_size = ensure_distributed_init()
    
    configs = CONFIGS
    if config_names:
        configs = {k: v for k, v in CONFIGS.items() if k in config_names}
        rprint(f"\nSelected configs: {list(configs.keys())}")
    
    # Warn about incompatible configs
    skippable = [n for n, c in configs.items() if c.tp_degree * c.ep_degree > world_size]
    if skippable:
        rprint(f"\n⚠ Configs requiring more workers than world_size={world_size}: {skippable}")
        rprint(f"  These will be skipped. Use torchrun --nproc_per_node=N for full benchmarks.")
    
    rprint("\nLoading input tensors...")
    runner = BenchmarkRunner(
        num_warmup=args.warmup,
        num_runs=args.runs,
        enable_profiling=args.profile,
    )
    inputs = runner.load_inputs()
    
    if not inputs:
        rprint("\nERROR: No input tensors found. Run 'python3 generate_inputs.py' first.")
        sys.exit(1)
    
    runner.run_all(configs, inputs)
    
    # Only rank 0 prints summary and saves — prevents multi-worker file overwrites
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        runner.print_summary_table()
        runner.save_results(args.output)


if __name__ == "__main__":
    main()
