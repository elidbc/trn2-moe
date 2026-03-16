# trn2-moe

CS217 Final Project — Profiling MoE layer kernels on AWS's Trainium2 accelerator.

## Setup

Run the environment launch script on a Trainium2 instance, and set the following env variable:
```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_training/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE="trn2"
```


Run the harness to execute custom kernels, use the following flags to specify what workload will be executed
```bash
python harness.py
```

### Profiling

To generate a profile:

To generate an .neff file when executing a kernel, ensure this env variable is set:
```bash
export NEURON_FRAMEWORK_DEBUG=1
```

To extract an .ntff profile file from a produced .neff file:

```bash
neuron-profile capture \
  -n ~/neuron_profiles/e2e_moe_20260311/e2e_moe.neff \
  -s ~/neuron_profiles/e2e_moe_20260311/e2e_moe.ntff
```

To view the CLI summary of the profiled Kernel:

```bash
neuron-profile view --output-format summary-text -n <path_to.neff> -s <path_to.ntff>
```

