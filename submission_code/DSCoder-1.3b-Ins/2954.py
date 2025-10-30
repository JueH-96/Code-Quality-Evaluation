from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, n + 1):
            dp[1][i] = max(dp[1][i - 1], prefix_sum[i] - prefix_sum[i - min(m, i)])
            for j in range(2, min(i + 1, m) + 1):
                dp[j][i] = max(dp[j][i - 1], dp[j - 1][i - 1] + max(prefix_sum[i] - prefix_sum[i - min(m, j - 1)], prefix_sum[i - min(m, j)] - prefix_sum[i - min(m, j - 1)]))

        return dp[m][n]