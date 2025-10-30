from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Compute prefix sums where prefix[i] = sum_{m=0}^{i-1} nums[m] * (-1)^m
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1] * (-1)**(i - 1)
        
        # Initialize variables
        max_even = 0  # dp[0] - prefix[0] = 0 - 0 = 0
        max_odd = float('-inf')
        prev_dp = 0  # dp[0] = 0
        
        # Iterate through the array to compute dp[i]
        for i in range(1, n + 1):
            # Compute dp[i]
            if max_odd == float('-inf'):
                dp = max(max_even + prefix[i], float('-inf'))
            else:
                dp = max(max_even + prefix[i], max_odd - prefix[i])
            
            # Update max_even or max_odd based on the parity of (i-1)
            if (i - 1) % 2 == 0:
                max_even = max(max_even, prev_dp - prefix[i - 1])
            else:
                max_odd = max(max_odd, prev_dp + prefix[i - 1])
            
            # Update previous dp value
            prev_dp = dp
        
        return dp