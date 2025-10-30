from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        prefix_sum = [0] * (n + 1)
        
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + nums[i - 1]
        
        for i in range(1, n + 1):
            dp[i][1] = prefix_sum[i]
        
        for x in range(2, k + 1):
            for i in range(1, n + 1):
                for j in range(1, i):
                    dp[i][x] = max(dp[i][x], dp[j][x - 1] + (prefix_sum[i] - prefix_sum[j]) * x)
        
        return dp[n][k]