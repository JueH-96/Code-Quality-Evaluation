from typing import List

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            start_k = (i + 1) // 2
            min_cost = float('inf')
            for k in range(start_k, i + 1):
                current = dp[k - 1] + prices[k - 1]
                if current < min_cost:
                    min_cost = current
            dp[i] = min_cost
        return dp[n]