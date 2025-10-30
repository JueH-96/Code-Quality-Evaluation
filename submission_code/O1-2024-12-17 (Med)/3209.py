class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        # dp[i] will store the minimum cost to acquire fruits 1 through i (1-indexed)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # No cost to acquire zero fruits
        
        for i in range(1, n + 1):
            # We only need to consider j where j + j >= i
            # which translates to j >= ceil(i/2).
            start_j = (i + 1) // 2  # smallest j to cover fruit i
            for j in range(start_j, i + 1):
                dp[i] = min(dp[i], dp[j - 1] + prices[j - 1])
        
        return dp[n]