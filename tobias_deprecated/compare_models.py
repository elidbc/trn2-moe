import os
import torch
import torch_neuronx

from mistral_moe import MixtralConfig, MixtralSparseMoeBlock
from mistral_moe_old import MixtralSparseMoeBlockOld

def check_close(tensor1, tensor2, rtol=1e-2, atol=1e-2):
    return torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

def main():
    config = MixtralConfig()
    model_path = "weights/model_weights/mistral_moe.pt"
    input_dir = "weights/input_weights"
    neuron_model_dir = "weights/compiled_model_weights"

    # Set up original uncompiled models
    print("Initializing Uncompiled MoE blocks...")
    moe_block_fast = MixtralSparseMoeBlock(config)
    moe_block_old = MixtralSparseMoeBlockOld(config)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        moe_block_fast.load_state_dict(state_dict)
        moe_block_old.load_state_dict(state_dict)
    else:
        print(f"Error: Model weights not found at {model_path}.")
        return

    # Cast uncompiled models to bfloat16 to match neuron compilation
    moe_block_fast = moe_block_fast.to(dtype=torch.bfloat16)
    moe_block_fast.eval()
    moe_block_old = moe_block_old.to(dtype=torch.bfloat16)
    moe_block_old.eval()

    inputs = [
        # "decode_b64_s1.pt",
        "prefill_b1_s4096.pt"
    ]

    for filename in inputs:
        input_path = os.path.join(input_dir, filename)
        if not os.path.exists(input_path):
            print(f"Skipping {filename}: not found.")
            continue
            
        print(f"\n--- Testing on {filename} ---")
        input_tensor = torch.load(input_path, map_location="cpu", weights_only=True).to(dtype=torch.bfloat16)
        
        # 1. Run inference on Native PyTorch Model (New/Fast)
        print("Running native PyTorch (Fast)...")
        with torch.no_grad():
            output_fast = moe_block_fast(input_tensor)

        # 2. Run inference on Native PyTorch Model (Old)
        print("Running native PyTorch (Old)...")
        with torch.no_grad():
            output_old = moe_block_old(input_tensor)

        # 3. Load and run appropriate Neuron Graph
        neuron_model_name = f"compiled_moe_{filename}.pt"
        neuron_model_path = os.path.join(neuron_model_dir, neuron_model_name)
        
        if os.path.exists(neuron_model_path):
            print(f"Loading and running Neuron model from {neuron_model_path}...")
            neuron_model = torch.jit.load(neuron_model_path)
            
            with torch.no_grad():
                output_neuron = neuron_model(input_tensor)
                
                # Neuron compile process outputs may be wrapped in a tuple if compiled a chunk that returned multiple outputs
                if isinstance(output_neuron, tuple):
                    output_neuron = output_neuron[0]
                elif isinstance(output_neuron, list):
                    output_neuron = output_neuron[0]
                elif hasattr(output_neuron, "items"): # Maybe a dict
                     output_neuron = list(output_neuron.values())[0]
        else:
            print(f"Neuron model not found at {neuron_model_path}. Skipping Neuron comparison.")
            output_neuron = None

        # Comparisons
        print("\nResults:")
        
        is_close_fast_old = check_close(output_fast, output_old)
        print(f"Fast vs Old Native Matches: {is_close_fast_old}")
        if not is_close_fast_old:
            max_diff = torch.max(torch.abs(output_fast - output_old))
            print(f"  Max diff: {max_diff.item():.4f}")

        if output_neuron is not None:
            is_close_fast_neuron = check_close(output_fast, output_neuron)
            print(f"Fast Native vs Neuron Matches: {is_close_fast_neuron}")
            if not is_close_fast_neuron:
                max_diff = torch.max(torch.abs(output_fast - output_neuron))
                print(f"  Max diff: {max_diff.item():.4f}")

            is_close_old_neuron = check_close(output_old, output_neuron)
            print(f"Old Native vs Neuron Matches: {is_close_old_neuron}")
            if not is_close_old_neuron:
                 max_diff = torch.max(torch.abs(output_old - output_neuron))
                 print(f"  Max diff: {max_diff.item():.4f}")

if __name__ == "__main__":
    main()
