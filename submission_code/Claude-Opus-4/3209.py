class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        # dp[i] = minimum cost to get all fruits from 0 to i (0-indexed)
        dp = [float('inf')] * n
        
        # Base case: we must buy the first fruit
        dp[0] = prices[0]
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
                
            # If we bought fruit at position i (0-indexed), 
            # we can take the next i+1 fruits for free
            # In 1-indexed terms: buying fruit at position i+1 
            # allows taking fruits at positions i+2 to 2*i+2
            for j in range(i + 1, min(n, i + 1 + i + 1)):
                dp[j] = min(dp[j], dp[i])
            
            # We can also choose to buy the next fruit
            if i + 1 < n:
                dp[i + 1] = min(dp[i + 1], dp[i] + prices[i + 1])
        
        return dp[n - 1]