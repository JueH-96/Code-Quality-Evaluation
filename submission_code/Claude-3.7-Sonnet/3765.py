from typing import List

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        
        # Precompute the prefix sums for nums and cost
        prefix_nums = [0] * (n + 1)
        prefix_cost = [0] * (n + 1)
        for i in range(n):
            prefix_nums[i+1] = prefix_nums[i] + nums[i]
            prefix_cost[i+1] = prefix_cost[i] + cost[i]
        
        # Initialize the DP table
        dp = [[float('inf')] * (n+1) for _ in range(n+1)]
        dp[0][0] = 0
        
        for i in range(1, n+1):
            for j in range(1, min(i+1, n+1)):
                for l in range(max(0, j-1), i):
                    subarray_cost = (prefix_nums[i] + k * j) * (prefix_cost[i] - prefix_cost[l])
                    dp[i][j] = min(dp[i][j], dp[l][j-1] + subarray_cost)
        
        return min(dp[n][j] for j in range(1, n+1))