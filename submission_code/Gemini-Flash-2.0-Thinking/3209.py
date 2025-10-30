class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            # Option 1: Purchase the i-th fruit
            dp[i] = min(dp[i], dp[i - 1] + prices[i - 1])

            # Option 2: The i-th fruit is obtained for free due to a previous purchase
            for j in range(1, i):
                if i > j and i <= j + j:
                    dp[i] = min(dp[i], dp[j])

        return dp[n]