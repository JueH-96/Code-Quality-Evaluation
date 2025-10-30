from typing import List
import math

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[k][i] represents the minimum cost to split the first i elements into k subarrays
        dp = [[0 for _ in range(n + 1)] for _ in range(4)]  # We use indices k=1,2,3 and i=1 to n
        
        # Base case for k=1: one subarray, cost is always nums[0]
        for i in range(1, n + 1):
            dp[1][i] = nums[0]
        
        # For k=2: split into two subarrays
        for i in range(2, n + 1):
            dp[2][i] = float('inf')  # Initialize to infinity
            for s in range(1, i):  # s ranges from 1 to i-1
                dp[2][i] = min(dp[2][i], dp[1][s] + nums[s])
        
        # For k=3: split into three subarrays
        for i in range(3, n + 1):
            dp[3][i] = float('inf')  # Initialize to infinity
            for s in range(2, i):  # s ranges from 2 to i-1
                dp[3][i] = min(dp[3][i], dp[2][s] + nums[s])
        
        return dp[3][n]