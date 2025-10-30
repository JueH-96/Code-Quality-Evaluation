class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(n):
            for j in range(i + 1, n + 1):
                dp[j] = min(dp[j], dp[i] + prices[i])

        return dp[n]