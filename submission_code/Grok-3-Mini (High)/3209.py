from typing import List

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            min_k = (i + 1) // 2
            dp[i] = min(dp[k - 1] + prices[k - 1] for k in range(min_k, i + 1))
        return dp[n]