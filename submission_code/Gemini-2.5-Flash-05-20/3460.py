import collections
from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7

        # Convert requirements list to a dictionary for O(1) lookup
        # Maps end_idx to required_inversion_count
        req_map = {end_i: cnt_i for end_i, cnt_i in requirements}

        # Find the maximum inversion count required by any requirement.
        # This will be the effective upper bound for our DP table size,
        # as we only care about inversion counts up to this maximum.
        # The problem constraints guarantee cnt_i <= 400.
        max_required_cnt = 0
        for _, cnt_i in requirements:
            max_required_cnt = max(max_required_cnt, cnt_i)
        
        # dp[j] will store the number of ways to form a permutation of 'k' elements
        # (conceptually, values 0 to k-1) such that it has 'j' inversions.
        # The size of dp array is max_required_cnt + 1.
        dp = [0] * (max_required_cnt + 1)
        
        # Base case: For 0 elements (empty prefix), there is 1 way (empty permutation)
        # that has 0 inversions.
        dp[0] = 1

        # Iterate k from 1 to n. 'k' represents the number of elements in the current prefix
        # being constructed (perm[0]...perm[k-1]).
        # In this DP formulation, we are conceptually inserting the value (k-1) into
        # permutations of (k-1) elements (values 0 to k-2).
        for k in range(1, n + 1):
            # Calculate the maximum possible inversions for 'k' elements.
            # This is k * (k - 1) / 2.
            # We cap the iteration for 'j_prime' (current inversion count) at the minimum of
            # max_required_cnt and the actual maximum inversions possible for 'k' elements.
            current_actual_max_inv = k * (k - 1) // 2
            loop_upper_bound_j_prime = min(max_required_cnt, current_actual_max_inv)

            new_dp = [0] * (max_required_cnt + 1) # Initialize the DP table for the next state (k elements)
            
            # Calculate prefix sums of the current dp array (which holds counts for k-1 elements).
            # This enables O(1) calculation for the sum of a range, using the sliding window approach.
            prefix_sum = [0] * (max_required_cnt + 1)
            prefix_sum[0] = dp[0]
            for j in range(1, max_required_cnt + 1):
                prefix_sum[j] = (prefix_sum[j-1] + dp[j]) % MOD

            # Populate new_dp (for k elements) based on dp (for k-1 elements).
            # new_dp[j'] = sum(dp[j' - new_inv_contrib]) where new_inv_contrib ranges from 0 to k-1.
            # This sum is equivalent to (prefix_sum[j'] - prefix_sum[j' - k]).
            for j_prime in range(0, loop_upper_bound_j_prime + 1):
                val = prefix_sum[j_prime]
                if j_prime - k >= 0:
                    # Subtract prefix_sum[j_prime - k]. Add MOD before % to handle negative results
                    val = (val - prefix_sum[j_prime - k] + MOD) % MOD 
                new_dp[j_prime] = val
            
            # Update the dp table for the current 'k' elements
            dp = new_dp

            # Check if current_end_idx (which is k-1) has a requirement.
            # If so, filter the dp table: zero out counts for all inversion numbers
            # that do not match the specific requirement.
            current_end_idx = k - 1
            if current_end_idx in req_map:
                required_cnt = req_map[current_end_idx]
                
                # Create a temporary array to hold the filtered state
                filtered_dp = [0] * (max_required_cnt + 1)
                # Only the count for 'required_cnt' should be kept.
                # Ensure required_cnt is within our DP table bounds (which it always should be due to max_required_cnt)
                if required_cnt <= max_required_cnt: 
                    filtered_dp[required_cnt] = dp[required_cnt]
                
                dp = filtered_dp
        
        # The problem guarantees that there is at least one requirement for end_i == n-1.
        # This means req_map[n-1] will always exist.
        # The final answer is the number of permutations of 'n' elements (perm[0...n-1])
        # that satisfy all requirements, and specifically the one for index n-1.
        # This count is stored in dp[req_map[n-1]].
        return dp[req_map[n-1]]