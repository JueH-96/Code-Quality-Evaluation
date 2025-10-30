from typing import List

class Solution:
  def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
    MOD = 10**9 + 7
    # Max possible cnt_i from constraints. This is the maximum number of inversions we need to track.
    MAX_TRACKED_INVERSIONS = 400 

    # Store requirements in a dictionary for quick lookup
    req_dict = {}
    for end_i, cnt_i in requirements:
        req_dict[end_i] = cnt_i

    # dp[j] will store the number of permutations of length `current_processing_length`
    # (using numbers 0...current_processing_length-1 relative to each other) 
    # that have exactly j inversions AND satisfy all requirements 
    # for prefixes ending at or before `current_processing_length-1`.
    
    # Initialize for length 0:
    # A permutation of 0 elements is empty, has 0 inversions, and there's 1 way to form it.
    dp = [0] * (MAX_TRACKED_INVERSIONS + 1)
    dp[0] = 1
    
    # Iterate for permutation length from 1 to n
    for current_len in range(1, n + 1):
        new_dp = [0] * (MAX_TRACKED_INVERSIONS + 1)
        
        # Calculate prefix sums of the dp table from the previous length (current_len - 1).
        # prefix_sum_prev_dp[k] = sum(dp[j] for j from 0 to k)
        prefix_sum_prev_dp = [0] * (MAX_TRACKED_INVERSIONS + 1)
        if MAX_TRACKED_INVERSIONS >=0: # dp[0] is valid only if array size is at least 1
             prefix_sum_prev_dp[0] = dp[0]
        for j in range(1, MAX_TRACKED_INVERSIONS + 1):
            prefix_sum_prev_dp[j] = (prefix_sum_prev_dp[j-1] + dp[j]) % MOD
        
        # Calculate new_dp for permutations of length `current_len`.
        # When forming a permutation of `current_len` elements by inserting the largest element (`current_len-1`)
        # into a permutation of `current_len-1` elements, it can create `k_new_inv` new inversions.
        # `k_new_inv` can range from 0 (inserted at the end) to `current_len-1` (inserted at the beginning).
        # So, new_dp[invs] = sum(dp[prev_invs]) where prev_invs is in [invs - (current_len-1), invs].
        # This sum is equivalent to prefix_sum_prev_dp[invs] - prefix_sum_prev_dp[invs - current_len].
        for invs in range(MAX_TRACKED_INVERSIONS + 1):
            # Sum of dp[j] for j from (invs - (current_len - 1)) to invs.
            # This is prefix_sum_prev_dp[invs] (sum from 0 to invs)
            # minus prefix_sum_prev_dp[invs - (current_len - 1) - 1] (sum from 0 to invs - current_len).
            
            current_sum_val = prefix_sum_prev_dp[invs]
            
            # Index of the prefix sum to subtract.
            # (invs - (current_len - 1) - 1) simplifies to (invs - current_len).
            idx_to_subtract_prefix_sum = invs - current_len 
            
            if idx_to_subtract_prefix_sum >= 0:
                current_sum_val = (current_sum_val - prefix_sum_prev_dp[idx_to_subtract_prefix_sum] + MOD) % MOD
            
            new_dp[invs] = current_sum_val
        
        dp = new_dp # Update dp table to represent permutations of current_len

        # Apply requirements if any for this current_len.
        # A permutation of `current_len` elements corresponds to prefix perm[0...current_len-1].
        # So, the relevant end_idx for requirements is `current_len-1`.
        end_idx_for_current_len = current_len - 1
        if end_idx_for_current_len in req_dict:
            expected_invs = req_dict[end_idx_for_current_len]
            
            # Constraint: 0 <= cnt_i <= 400, so expected_invs is within [0, MAX_TRACKED_INVERSIONS].
            # Filter dp table: only entries matching expected_invs are kept.
            for j in range(MAX_TRACKED_INVERSIONS + 1):
                if j != expected_invs:
                    dp[j] = 0
                            
    # The problem guarantees a requirement for end_i = n-1.
    # The final dp array is for permutations of length n.
    # The requirement for end_i = n-1 (which means current_len = n) has been applied in the last iteration.
    # So, dp[req_dict[n-1]] will hold the answer. All other dp[j] would have been zeroed out.
    
    # final_expected_invs is req_dict[n-1]. This key is guaranteed to exist.
    # Its value (cnt_i) is also guaranteed to be within [0, MAX_TRACKED_INVERSIONS].
    return dp[req_dict[n-1]]