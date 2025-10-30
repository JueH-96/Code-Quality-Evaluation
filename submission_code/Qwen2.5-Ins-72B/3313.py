from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[float('-inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                dp[i][j] = dp[i - 1][j]
                sign = 1 if j % 2 == 1 else -1
                for start in range(i):
                    current_sum = sum(nums[start:i])
                    dp[i][j] = max(dp[i][j], dp[start][j - 1] + sign * current_sum * (k - j + 1))
        
        return dp[n][k]