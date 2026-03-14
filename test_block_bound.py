import math
import itertools
from functools import lru_cache

def calculate_theoretical_bound(num_tokens, top_k, num_experts, block_size):
    """The O(1) mathematical bound for the maximum possible blocks needed."""
    
    # 1. The Fragmentation Bound (accounting for ghost experts)
    total_routed_tokens = num_tokens * top_k
    # An expert needs at least 1 token to cause padding. 
    # You can't have more active experts than total tokens.
    active_experts = min(num_experts, total_routed_tokens)
    bound_1 = math.floor((total_routed_tokens + (active_experts * (block_size - 1))) / block_size)
    
    # 2. The Pigeonhole Bound (accounting for maximum capacity per expert)
    # An expert can never receive more than 'num_tokens' because it can only be chosen once per token.
    # Therefore, the maximum blocks an individual expert can ever require is ceil(num_tokens / block_size).
    bound_2 = num_experts * math.ceil(num_tokens / block_size)
    
    # The true mathematical worst-case is the stricter of the two limits
    return min(bound_1, bound_2)

def calculate_brute_force_bound(num_tokens, top_k, num_experts, block_size):
    """
    Uses Dynamic Programming to test every valid distribution of tokens to experts,
    finding the exact configuration that forces the maximum number of blocks.
    """
    total_routed_tokens = num_tokens * top_k

    # A fresh cache is automatically created for this nested function on every call
    @lru_cache(None)
    def search(expert_idx, remaining_tokens):
        # Base Case: All experts have been assigned tokens
        if expert_idx == num_experts:
            # If we perfectly assigned all tokens, this is a valid distribution
            return 0 if remaining_tokens == 0 else -float('inf')

        max_buckets_for_branch = -float('inf')
        
        # An expert cannot receive more tokens than the total sequence length (T),
        # because an expert can only be chosen a maximum of once per token.
        max_possible_for_expert = min(num_tokens, remaining_tokens)

        # Pruning: If the remaining experts couldn't possibly absorb the remaining tokens
        # even if they all took their maximum allowance, this branch is mathematically invalid.
        if remaining_tokens > (num_experts - expert_idx) * num_tokens:
            return -float('inf')

        # Brute force check every possible number of tokens this expert could receive
        for c in range(max_possible_for_expert + 1):
            buckets_for_this_expert = math.ceil(c / block_size)
            
            # Recurse and add to find the maximum possible buckets down this branch
            result = buckets_for_this_expert + search(expert_idx + 1, remaining_tokens - c)
            if result > max_buckets_for_branch:
                max_buckets_for_branch = result

        return max_buckets_for_branch

    return search(0, total_routed_tokens)

def run_test_suite():
    # Define our hyperparameter grids to brute force
    num_tokens_list = [5, 10, 32, 64, 128]
    top_k_list = [1, 2, 4]
    num_experts_list = [4, 8, 16]
    block_sizes = [2, 4, 16, 32]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = [] # Track our failures

    print("Running MoE BWMM Maximum Bounds Test Suite...")
    print("-" * 65)
    print(f"{'T (Tokens)':<12} | {'K (Top-K)':<10} | {'E (Experts)':<12} | {'B (Block Size)':<15} | {'Match?'}")
    print("-" * 65)

    # Iterate through every combination of parameters
    for T, K, E, B in itertools.product(num_tokens_list, top_k_list, num_experts_list, block_sizes):
        
        # Skip invalid MoE configurations (e.g., trying to route top-4 when only 2 experts exist)
        if K > E:
            continue
            
        total_tests += 1
        
        # Calculate both bounds
        theoretical_max = calculate_theoretical_bound(T, K, E, B)
        brute_force_max = calculate_brute_force_bound(T, K, E, B)
        
        # Check if they match
        is_match = theoretical_max == brute_force_max
        if is_match:
            passed_tests += 1
        else:
            # Store the failure data for the final report
            failed_tests.append({
                'T': T, 'K': K, 'E': E, 'B': B, 
                'Theory': theoretical_max, 'Brute': brute_force_max
            })
            
        # Print a subset of passing tests to show progress, but explicitly print ALL failures
        if total_tests % 15 == 0 or not is_match:
            match_str = "✅ YES" if is_match else f"❌ NO (Theory: {theoretical_max}, Brute: {brute_force_max})"
            print(f"{T:<12} | {K:<10} | {E:<12} | {B:<15} | {match_str}")

    # Generate the final report
    print("-" * 65)
    print(f"Test Suite Completed: {passed_tests}/{total_tests} tests passed.")
    
    if not failed_tests:
        print("The theoretical Trainium2 padding equation is mathematically airtight. Zero failures found.")
    else:
        print(f"\n⚠️ WARNING: {len(failed_tests)} tests failed the boundary check!")
        print("Failed Configurations Report:")
        for fail in failed_tests:
             print(f"  -> T={fail['T']}, K={fail['K']}, E={fail['E']}, B={fail['B']} | "
                   f"Theory Bound: {fail['Theory']} | Actual Brute Force: {fail['Brute']}")

if __name__ == "__main__":
    run_test_suite()