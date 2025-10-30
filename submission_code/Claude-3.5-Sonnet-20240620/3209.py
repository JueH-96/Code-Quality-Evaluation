class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [float('inf')] * (n + 1)
        dp[n] = 0
        
        for i in range(n - 1, -1, -1):
            cost = prices[i]
            free_fruits = min(i + 1, n - i)
            dp[i] = cost + min(dp[i+1:i+free_fruits+1])
        
        return dp[0]