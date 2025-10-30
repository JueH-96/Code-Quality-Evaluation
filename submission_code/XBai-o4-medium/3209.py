from typing import List

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [0] * (n + 1)  # dp[0 ... n]
        for i in range(1, n + 1):
            start_j = (i + 1) // 2  # starting j (1-based) for current i
            min_val = float('inf')
            # Iterate from start_j to i (inclusive)
            for j in range(start_j, i + 1):
                current = dp[j - 1] + prices[j - 1]
                if current < min_val:
                    min_val = current
            dp[i] = min_val
        return dp[n]