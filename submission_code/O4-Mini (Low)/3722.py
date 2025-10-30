from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        # prefix sums
        pre = [0] * (n + 1)
        for i in range(1, n + 1):
            pre[i] = pre[i - 1] + nums[i - 1]
        # dp[i][j]: max sum using i segments in first j elements
        NEG_INF = -10**18
        dp = [[NEG_INF] * (n + 1) for _ in range(k + 1)]
        # zero segments gives sum 0
        for j in range(n + 1):
            dp[0][j] = 0
        
        for i in range(1, k + 1):
            best_prev = NEG_INF
            # we build dp[i][*]
            for j in range(0, n + 1):
                # first, update best_prev using position j-m
                if j - m >= 0:
                    val = dp[i-1][j-m] - pre[j-m]
                    if val > best_prev:
                        best_prev = val
                # then compute dp[i][j]
                if j == 0:
                    dp[i][j] = NEG_INF
                else:
                    take = NEG_INF
                    if best_prev > NEG_INF:
                        take = pre[j] + best_prev
                    # either we don't end a segment at j, or we do
                    dp[i][j] = max(dp[i][j-1], take)
        
        return dp[k][n]