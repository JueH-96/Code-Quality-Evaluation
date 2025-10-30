from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        # Initialize a large value for dp array
        dp = [[float('inf')] * 4 for _ in range(n + 1)]
        dp[0][0] = 0

        # Fill the dp array
        for i in range(1, n + 1):
            for j in range(1, 4):
                dp[i][j] = min(dp[i][j], dp[i - 1][j] + nums[i - 1])

        # Calculate the minimum cost
        min_cost = float('inf')
        for i in range(2, n):
            for j in range(i + 1, n + 1):
                cost = dp[i][1] + dp[j][2] - dp[i][2] + dp[n][3] - dp[j][3]
                min_cost = min(min_cost, cost)

        return min_cost