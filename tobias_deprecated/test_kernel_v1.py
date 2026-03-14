import torch
from kernel_v1 import create_inputs
from mistral_moe import MixtralConfig

def test_create_inputs():
    print("Testing create_inputs in kernel_v1.py...")
    config = MixtralConfig()
    
    # Example dimensions: batch_size=2, seq_len=4, hidden_dim=config.hidden_dim (e.g., 4096)
    batch_size = 2
    seq_len = 400
    hidden_dim = config.hidden_dimension if hasattr(config, 'hidden_dimension') else 4096
    
    print(f"Creating input tensor with shape: {(batch_size, seq_len, hidden_dim)}")
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
    
    try:
        outputs = create_inputs(input_tensor)
        print("Success! create_inputs ran without errors.")
        print(f"Returned {len(outputs)} tensors:")
        
        names = [
            "padded_sorted_tokens",
            "w1_experts",
            "w3_experts",
            "w2_experts",
            "padded_routing_weights",
            "expert_offsets",
            "output_tokens"
        ]
        
        for i, out in enumerate(outputs):
            print(f"  {names[i]} -> shape: {out.shape}, dtype: {out.dtype}")
            if names[i] == "expert_offsets":
                print(out)
            
    except Exception as e:
        print("Error during create_inputs execution!")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_create_inputs()
