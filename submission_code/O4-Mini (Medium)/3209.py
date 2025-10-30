from typing import List

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        # dp[i] = minimum coins to cover fruits 1..i
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        # For each purchase at position j (1-based), interval covers [j, j + j]
        for j in range(1, n + 1):
            cost = prices[j - 1]
            right = min(n, j + j)
            # If we buy at j, we must have covered up to j-1,
            # and then this purchase covers j..right for free.
            base = dp[j - 1] + cost
            for i in range(j, right + 1):
                if dp[i] > base:
                    dp[i] = base
        
        return dp[n]