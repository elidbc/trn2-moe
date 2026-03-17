import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import glob
import torch_neuronx # Required for Trainium compilation
from mistral_moe import MixtralConfig, MixtralSparseMoeBlock

INPUT_MODEL_PATH = "../weights/uncompiled_model_weights/mistral_moe_new.pt"
INPUT_TENSOR_PATH = "../weights/input_weights/prefill_b1_s4096.pt"
OUTPUT_MODEL_PATH = "../weights/compiled_model_weights/mistral_moe_new.pt"
OUTPUT_TENSOR_PATH = "../weights/output_weights/out_prefill_b1_s4096.pt"
BLOCK_SIZE = 128


def main():
    print("Initializing MoE block...")
    config = MixtralConfig()
    moe_block = MixtralSparseMoeBlock(config)
    
    if os.path.exists(INPUT_MODEL_PATH):
        print(f"Loading model weights from '{INPUT_MODEL_PATH}'...")
        state_dict = torch.load(INPUT_MODEL_PATH, map_location="cpu", weights_only=True)
        moe_block.load_state_dict(state_dict)
    else:
        print(f"Error: Model weights not found at {INPUT_MODEL_PATH}.")
        return
        
    # Cast to bfloat16 and set to eval mode
    moe_block = moe_block.to(dtype=torch.bfloat16)
    moe_block.eval()
    
    with torch.no_grad(): 
        # Load input and cast to bfloat16
        input_tensor = torch.load(INPUT_TENSOR_PATH, map_location="cpu", weights_only=True)
        input_tensor = input_tensor.to(dtype=torch.bfloat16)
        
        # 1. Trace the model if we haven't seen this shape yet
        print(f"--- Tracing Neuron graph for shape {input_tensor.shape} ---")
            
        # torch_neuronx.trace compiles the model specifically for this tensor shape
        compiled_model = torch_neuronx.trace(moe_block, input_tensor)
        print("--- Tracing complete! ---\n")
        
        # save compiled model 
        print(f"Saving compiled model locally to {OUTPUT_MODEL_PATH}...")
        torch.jit.save(compiled_model, OUTPUT_MODEL_PATH)
        
        # compute output tensor
        output_tensor, selected_experts = compiled_model(input_tensor)
        
        # save output tensor
        print(f"Saving output tensor locally to {OUTPUT_TENSOR_PATH}...")
        torch.save(output_tensor, OUTPUT_TENSOR_PATH)


if __name__ == "__main__":
    main()