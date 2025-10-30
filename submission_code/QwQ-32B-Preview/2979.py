from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort the offers based on the end indices
        sorted_offers = sorted(offers, key=lambda x: x[1])
        
        # Initialize dp array with 0s, with dp[0] = 0
        dp = [0] * (n + 1)
        
        # Iterate through each offer in the sorted list
        for start, end, gold in sorted_offers:
            # Update dp[end+1] to be the maximum of its current value and dp[start] + gold
            dp[end + 1] = max(dp[end + 1], dp[start] + gold)
        
        # The maximum profit is in dp[n]
        return dp[n]