import math
from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 1000000007
        N = len(nums)
        dp = [0] * (N + 1)
        
        # Initialize dp array
        if N >= 1:
            dp[1] = max(0, nums[0])
            for i in range(2, N + 1):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        
        ans = 0
        
        for query in queries:
            pos, x = query[0], query[1]
            nums[pos] = x  # Update the nums array
            
            start = pos + 1  # Start index to recompute dp
            
            if start <= N:
                for i in range(start, N + 1):
                    if i == 1:
                        dp[i] = max(0, nums[0])
                    else:  # i >= 2
                        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
            
            # Add the current maximum sum to ans with modulo
            ans = (ans + dp[N]) % MOD
        
        return ans