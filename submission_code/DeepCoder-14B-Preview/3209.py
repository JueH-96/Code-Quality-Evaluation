class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        if n == 0:
            return 0
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 coins needed to cover 0 fruits
        
        for j in range(n):
            current_max_reach = min(2 * (j + 1), n)
            for i in range(j + 1, current_max_reach + 1):
                if dp[i] > dp[j] + prices[j]:
                    dp[i] = dp[j] + prices[j]
        
        return dp[n]