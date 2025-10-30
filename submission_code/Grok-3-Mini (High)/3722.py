import math
from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        # Compute prefix sum
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        
        # Define a large negative number for impossible states
        NEG_INF = -30000000  # Min possible sum is -10^4 * 2000 = -2e7, so this is safe
        
        # dp[i][sub] : max sum with first i elements and exactly sub subarrays
        dp = [[NEG_INF for _ in range(k + 1)] for _ in range(n + 1)]
        
        # Base case: with 0 subarrays, sum is 0
        for i in range(n + 1):
            dp[i][0] = 0
        
        # Fill dp table
        for num_sub in range(1, k + 1):  # for each number of subarrays
            low = (num_sub - 1) * m  # Minimum elements for num_sub - 1 subarrays
            # Compute val[start] = dp[start][num_sub - 1] - prefix[start]
            val = [dp[start][num_sub - 1] - prefix[start] for start in range(n + 1)]
            
            # Compute dp using cumulative max optimization
            current_max_val = NEG_INF
            for high_val in range(low, n - m + 1):  # high_val is the upper limit for start index
                # Update the current maximum value
                current_max_val = max(current_max_val, val[high_val])
                # i is the number of elements, i = high_val + m
                i = high_val + m
                # dp[i][num_sub] = prefix[i] + max over start of (dp[start][num_sub-1] - prefix[start])
                dp[i][num_sub] = prefix[i] + current_max_val
        
        # The answer is dp[n][k], with all n elements and exactly k subarrays
        return dp[n][k]