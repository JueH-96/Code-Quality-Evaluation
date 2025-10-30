from typing import List
import bisect

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort offers based on end_i
        offers.sort(key=lambda x: x[1])
        
        # dp array to store the maximum profit up to each house
        dp = [0] * (n + 1)
        
        for start, end, gold in offers:
            # Find the maximum profit up to the start of the current offer
            max_profit_before_start = dp[start]
            # Calculate the profit if we take the current offer
            current_profit = max_profit_before_start + gold
            # Update the dp array from end+1 onwards
            dp[end + 1] = max(dp[end + 1], current_profit)
        
        # Propagate the maximum profit to the end of the dp array
        for i in range(1, n + 1):
            dp[i] = max(dp[i], dp[i - 1])
        
        return dp[n]