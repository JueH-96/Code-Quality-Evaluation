class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [float('inf')] * (n + 1)
        dp[n] = 0
        for i in range(n - 1, -1, -1):
            for j in range(i, min(2 * i + 2, n)):
                dp[i] = min(dp[i], prices[i] + dp[j + 1])
        return dp[0]