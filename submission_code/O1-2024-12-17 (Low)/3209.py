class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        
        # dp[j] will store the minimum cost to acquire fruits 0 through j (inclusive).
        dp = [float('inf')] * n
        dp[0] = prices[0]
        
        for j in range(1, n):
            # Try buying some fruit i (where i <= j) that covers up to j
            # Coverage condition: if you buy fruit i, you cover i..i+(i+1) in 0-based indexing
            # So we need i+(i+1) >= j  ->  2*i+1 >= j
            for i in range(j + 1):
                if 2*i + 1 >= j:
                    cost_before = dp[i-1] if i > 0 else 0
                    dp[j] = min(dp[j], cost_before + prices[i])
        
        return dp[n-1]