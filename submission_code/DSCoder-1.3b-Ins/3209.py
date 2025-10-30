class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        dp = [0] * (len(prices) + 1)
        for i in range(len(prices) - 1, -1, -1):
            if prices[i] > prices[i + 1]:
                prices[i] -= prices[i + 1]
                dp[i] += 1
            dp[i] += dp[i + 1]
        return min(dp)