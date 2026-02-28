import numpy as np
import torch
import torch.nn.functional as F

NUM_EXPERTS = 8
TOP_K = 2

def moe_pytorch(x, w, router_weights):
    batch_dim, seq_dim, feature_dim = x.shape
    num_experts, feature_dim, expert_dim = w.shape

    # router
    x_flat = x.reshape(batch_dim * seq_dim, feature_dim)
    token_logits = x_flat @ router_weights
    top_k_indices = torch.argsort(token_logits, dim=-1, descending=True)[:, :TOP_K]
    top_k_logits = torch.take_along_axis(token_logits, top_k_indices, axis=-1) # shape (batch_dim * seq_dim, TOP_K)
    top_k_probs = F.softmax(top_k_logits, dim=-1) # shape (batch_dim * seq_dim, TOP_K)

    # experts
    return
    
    

def host_code():
    batch_dim = 1024
    seq_dim = 1024
    feature_dim = 4096
    expert_dim = 14336

    # router: flattens activations to (batch_dim * seq_dim, feature_dim)
    # router weights: (feature_dim, NUM_EXPERTS)

    x = np.random.rand(batch_dim, seq_dim, feature_dim)

    # 8 experts, each with w1 (gate), w3 (up), w2 (down)
    w = [np.random.rand(3, feature_dim, expert_dim) for _ in range(NUM_EXPERTS)] 
    router_weights = np.random.rand(feature_dim, NUM_EXPERTS)
    return