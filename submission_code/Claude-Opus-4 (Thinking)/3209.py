class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        # dp[i] = minimum cost to acquire fruits 1 to i
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # base case: no cost for 0 fruits
        
        for i in range(1, n + 1):
            # Try purchasing fruit j to cover fruit i
            for j in range(1, i + 1):
                # Check if purchasing fruit j covers fruit i
                if j + j >= i:
                    # Cost = cost to get fruits 1 to j-1 + cost of fruit j
                    dp[i] = min(dp[i], dp[j - 1] + prices[j - 1])
        
        return dp[n]