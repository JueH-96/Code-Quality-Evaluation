class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        # dp[i] will store the minimum coins needed to purchase all fruits from i to n-1
        dp = [float('inf')] * (n + 1)
        dp[n] = 0  # No cost to purchase beyond the last fruit
        
        # Fill dp array from the end to the start
        for i in range(n - 1, -1, -1):
            # Calculate the cost if we buy the ith fruit and get the next i fruits for free
            sum_cost = 0
            for j in range(i, min(n, i + i + 1)):
                sum_cost += prices[j]
                dp[i] = min(dp[i], sum_cost + dp[j + 1])
        
        return dp[0]