class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [float('inf')] * (n + 1)
        dp[n] = 0
        
        for i in range(n-1, -1, -1):
            curr_cost = prices[i]
            free_fruits = i + 1
            max_free = min(n, free_fruits + i + 1)
            
            dp[i] = curr_cost + min(dp[j] for j in range(max_free + 1, n + 1))
            
        return dp[0]