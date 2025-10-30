class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            dp[i] = prices[i] + dp[min(n, i + i + 1)]
            if i + 1 < n:
                dp[i] = min(dp[i], dp[i + 1])
        return dp[0]