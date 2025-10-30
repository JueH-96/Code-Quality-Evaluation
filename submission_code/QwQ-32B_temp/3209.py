class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            start_j = (i + 1) // 2  # Calculate the starting j as ceil(i/2)
            min_cost = float('inf')
            for j in range(start_j, i + 1):
                current_cost = dp[j - 1] + prices[j - 1]
                if current_cost < min_cost:
                    min_cost = current_cost
            dp[i] = min_cost
        return dp[n]