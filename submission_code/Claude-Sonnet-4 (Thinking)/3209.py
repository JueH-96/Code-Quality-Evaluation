class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [float('inf')] * n
        
        for i in range(n):
            for j in range(max(0, i // 2), i + 1):
                cost = prices[j] if j == 0 else dp[j - 1] + prices[j]
                dp[i] = min(dp[i], cost)
        
        return dp[n - 1]