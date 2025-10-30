class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        # dp[i] represents minimum cost to get all fruits from index i onwards
        dp = [0] * (n + 1)
        
        # Work backwards
        for i in range(n - 1, -1, -1):
            # Buy fruit i (0-indexed), which gives us fruits i+1 to i+(i+1) for free
            # So next fruit we need to consider is at index i+(i+1)+1 = i+i+2
            next_buy = min(n, i + i + 2)
            dp[i] = prices[i] + dp[next_buy]
            
            # But we might get fruit i for free from buying an earlier fruit
            # If so, we just need dp[i+1]
            if i < n - 1:
                dp[i] = min(dp[i], dp[i + 1])
        
        return dp[0]