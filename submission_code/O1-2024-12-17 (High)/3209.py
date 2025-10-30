class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        # dp[i] will store the minimum cost to acquire fruits 1 through i (1-indexed)
        dp = [0] + [float('inf')] * n
        
        for i in range(1, n + 1):
            # We only need to consider j such that buying fruit j covers up to fruit i
            # i.e., if we buy fruit j, we cover j through j + j, so we need j + j >= i
            # => j >= i/2
            start_j = (i + 1) // 2  # smallest j that can cover i
            for j in range(start_j, i + 1):
                # dp[j-1] is the cost to cover up to fruit j-1
                # plus prices[j-1] is the cost to buy fruit j (1-indexed vs 0-indexed)
                dp[i] = min(dp[i], dp[j - 1] + prices[j - 1])
        
        return dp[n]