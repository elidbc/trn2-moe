import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import glob
import torch_neuronx # Required for Trainium compilation
import boto3 # Added for S3 upload
from mistral_moe import MixtralConfig, MixtralSparseMoeBlock

def main():
    config = MixtralConfig()
    model_path = "../weights/uncompiled_model_weights/mistral_moe.pt"
    input_dir = "../weights/input_weights"
    output_dir = "../weights/output_weights"
    
    # --- S3 Configuration ---
    bucket_name = "cs217-moe-neuron-cache" # REPLACE THIS with your actual bucket name
    s3_prefix = "compiled_moe_models"
    s3_client = boto3.client('s3')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing MoE block...")
    moe_block = MixtralSparseMoeBlock(config)
    
    if os.path.exists(model_path):
        print(f"Loading model weights from '{model_path}'...")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        moe_block.load_state_dict(state_dict)
    else:
        print(f"Error: Model weights not found at {model_path}.")
        print("Please run benchmark.py first to generate the model weights.")
        return
        
    # Cast to bfloat16 (Trainium's native optimized format)
    moe_block = moe_block.to(dtype=torch.bfloat16)
    moe_block.eval()
    

    # overwrite input files to restrict the number of NEFF files
    input_files = [f"{input_dir}/decode_b64_s1.pt", f"{input_dir}/prefill_b1_s4096.pt"]
    compiled_model = None
    
    with torch.no_grad(): 
        for input_path in input_files:
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, f"out_{filename}")
            
            # Load input and cast to bfloat16
            input_tensor = torch.load(input_path, map_location="cpu", weights_only=True)
            input_tensor = input_tensor.to(dtype=torch.bfloat16)
            
            # Extract the shape to use as a dictionary key
            shape_key = tuple(input_tensor.shape)
            
            # 1. Trace the model if we haven't seen this shape yet
            print(f"--- Tracing Neuron graph for shape {shape_key} ---")
            print("This will take a few minutes if not found in the local cache...")
                
            # torch_neuronx.trace compiles the model specifically for this tensor shape
            compiled_model = torch_neuronx.trace(moe_block, input_tensor)
            print("--- Tracing complete! ---\n")
            
            # --- NEW: Save locally, upload to S3, and cleanup ---
            # Create a unique filename based on the tensor shape (e.g., compiled_moe_64_1_4096.pt)
            shape_str = "_".join(map(str, shape_key))
            local_save_name = f"compiled_moe_{filename}.pt"
            s3_key = f"{s3_prefix}/{local_save_name}"
            
            print(f"Saving compiled model locally to {local_save_name}...")
            torch.jit.save(compiled_model, local_save_name)
            
            print(f"Uploading to s3://{bucket_name}/{s3_key}...")
            s3_client.upload_file(local_save_name, bucket_name, s3_key)
            print("Upload complete!")
            
            print(f"Processing '{filename}' | Shape: {list(input_tensor.shape)}...")
            
            # Note: The output is a tuple (final_hidden_states, router_logits) based on our batched block
            # If your module just returns the tensor, remove the [0]
            output_tensor = compiled_model(input_tensor)
            
            # 3. Save output to disk
            torch.save(output_tensor.cpu(), output_path)
            print(f" -> Saved output to 'out_{filename}' | Shape: {list(output_tensor.shape)}\n")
            
    print("Successfully processed all inputs through the Neuron compiler!")

if __name__ == "__main__":
    main()