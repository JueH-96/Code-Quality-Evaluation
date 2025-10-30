from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        prefix_sum = [0] * (n + 1)
        
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + nums[i - 1]
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                dp[i][j] = max(dp[i - 1][j], 
                               prefix_sum[i] - prefix_sum[max(0, i - j)] * (k - j + 1) + dp[max(0, i - j)][j - 1])
        
        return dp[n][k]