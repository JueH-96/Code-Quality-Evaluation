from typing import List

MOD = 10**9 + 7

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize for the first element (i=0)
        prev_max = nums[0]
        dp_prev = [1] * (prev_max + 1)  # dp_prev[a] is 1 for all a in 0..prev_max
        
        # Compute prefix sums for the initial dp_prev
        prefix_prev = [0] * (prev_max + 1)
        prefix_prev[0] = dp_prev[0]
        for j in range(1, prev_max + 1):
            prefix_prev[j] = (prefix_prev[j-1] + dp_prev[j]) % MOD
        
        # Iterate from i=1 to n-1
        for i in range(1, n):
            current_max = nums[i]
            delta = max(0, nums[i] - nums[i-1])
            
            # Initialize current_dp
            current_dp = [0] * (current_max + 1)
            
            for a in range(current_max + 1):
                upper_b = a - delta
                upper_b = min(upper_b, nums[i-1])  # Ensure upper_b does not exceed previous max
                
                if upper_b < 0:
                    current_dp[a] = 0
                else:
                    # Ensure upper_b is within the bounds of prefix_prev
                    current_dp[a] = prefix_prev[upper_b] % MOD
            
            # Compute prefix sums for current_dp
            prefix_current = [0] * (current_max + 1)
            prefix_current[0] = current_dp[0]
            for j in range(1, current_max + 1):
                prefix_current[j] = (prefix_current[j-1] + current_dp[j]) % MOD
            
            # Update for next iteration
            dp_prev, prefix_prev = current_dp, prefix_current
        
        # Sum all possible values in the last dp_prev
        return sum(dp_prev) % MOD