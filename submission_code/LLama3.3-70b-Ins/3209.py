from typing import List

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [0] * (n + 1)
        
        for i in range(n - 1, -1, -1):
            # Calculate the minimum coins needed if we purchase the i-th fruit
            purchase = prices[i] + dp[min(i + i + 1, n)]
            
            # Calculate the minimum coins needed if we don't purchase the i-th fruit
            # We can only do this if we have already purchased a fruit that allows us to take the i-th fruit for free
            if i > 0:
                not_purchase = dp[i + 1]
            else:
                not_purchase = float('inf')
            
            # Choose the option that requires the minimum number of coins
            dp[i] = min(purchase, not_purchase)
        
        return dp[0]