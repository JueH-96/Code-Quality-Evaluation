class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            dp[i] = min(dp[i], dp[i-1] + prices[i-1])
            for j in range(i + 1, min(n+1, i + i + 1)):
                dp[j] = min(dp[j], dp[i-1] + prices[i-1])
        
        return dp[n]