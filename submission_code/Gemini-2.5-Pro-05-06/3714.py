from typing import List

class Solution:
  def minMaxSums(self, nums: List[int], k: int) -> int:
    MOD = 10**9 + 7
    
    n = len(nums)
    nums.sort() # s_nums is now just nums
    
    # Precompute modular inverses for 1, ..., k-1
    # inv[j] stores j^(-1) mod MOD
    # Max idx for inv[idx] is k-1. inv array size k is sufficient (indices 0 to k-1).
    # inv[0] is not used. inv[1] is 1.
    inv = [1] * k 
    # Loop computes inv[2]...inv[k-1]
    for i in range(2, k): 
        inv[i] = pow(i, MOD - 2, MOD)

    memo_S_M_R = {}
    # Function to calculate sum_{idx=0 to R_limit} (M C idx) % MOD
    def calculate_sum_C(M: int, R_limit: int) -> int:
        state = (M, R_limit)
        if state in memo_S_M_R:
            return memo_S_M_R[state]

        # M C 0 = 1. This is the first term.
        current_term_C_M_idx = 1 
        current_sum = 1
        
        # Loop for idx from 1 up to min(R_limit, M)
        # C(M, idx) = C(M, idx-1) * (M - idx + 1) / idx
        # current_term_C_M_idx holds C(M, idx-1) at the start of an iteration for idx.
        # It's updated to C(M, idx) within the iteration.
        for idx in range(1, min(R_limit, M) + 1):
            numerator = M - idx + 1 # This is (M - (idx-1)) part of M C idx formula if starting from M C (idx-1)
            
            current_term_C_M_idx = (current_term_C_M_idx * numerator) % MOD
            current_term_C_M_idx = (current_term_C_M_idx * inv[idx]) % MOD # inv[idx] is idx^(-1)
            
            current_sum = (current_sum + current_term_C_M_idx) % MOD
        
        memo_S_M_R[state] = current_sum
        return current_sum

    total_min_sum = 0
    # R_limit for combinations is k-1 (subsequences of length 1 to k means choosing 0 to k-1 additional elements)
    R_val_for_sum = k - 1

    # If k=0, this would be -1. Problem states k is positive (1 <= k).
    # If k=1, R_val_for_sum = 0. calculate_sum_C(M,0) will correctly return 1.

    for i in range(n):
        # nums[i] is min, other elements from nums[i+1...n-1]
        # Number of available elements to choose from is (n-1-i)
        M_val = n - 1 - i
        comb_sum_val = calculate_sum_C(M_val, R_val_for_sum)
        term = (nums[i] * comb_sum_val) % MOD
        total_min_sum = (total_min_sum + term) % MOD
            
    total_max_sum = 0
    for i in range(n):
        # nums[i] is max, other elements from nums[0...i-1]
        # Number of available elements to choose from is i
        M_val = i
        comb_sum_val = calculate_sum_C(M_val, R_val_for_sum)
        term = (nums[i] * comb_sum_val) % MOD
        total_max_sum = (total_max_sum + term) % MOD
            
    return (total_min_sum + total_max_sum) % MOD