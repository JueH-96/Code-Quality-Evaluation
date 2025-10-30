class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        dp = [float('inf')] * (len(prices) + 1)
        dp[0] = 0
        
        for i in range(len(prices)):
            dp[i + 1] = min(dp[i + 1], prices[i] + dp[max(0, i - i)])  # Buy current fruit i
            for j in range(1, i + 2):  # Take j fruits for free starting from the current one
                dp[min(len(prices), i + j + 1)] = min(dp[min(len(prices), i + j + 1)], dp[i] + prices[i])
        
        return dp[-1]