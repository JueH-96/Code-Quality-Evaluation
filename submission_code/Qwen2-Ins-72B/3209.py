class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [float('inf')] * n
        dp[-1] = prices[-1]
        for i in range(n-2, -1, -1):
            dp[i] = prices[i] + min(dp[i+1:i+1+i+1] + [0])
        return dp[0]