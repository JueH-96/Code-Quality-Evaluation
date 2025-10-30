class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 coins needed to acquire 0 fruits
        
        for j in range(1, n + 1):
            cost = prices[j - 1]
            end = min(j + j, n)
            # Update all positions from j to end with the potential new minimum cost
            for k in range(j, end + 1):
                if dp[j - 1] + cost < dp[k]:
                    dp[k] = dp[j - 1] + cost
        
        return dp[n]