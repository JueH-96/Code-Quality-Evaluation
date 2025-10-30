class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [0] * (n + 1)
        
        for i in range(n - 1, -1, -1):
            take_free = dp[i + 1] if i + 1 < n else 0
            take_purchase = prices[i] + dp[min(i + i + 1, n)]
            dp[i] = min(take_free, take_purchase)
        
        return dp[0]