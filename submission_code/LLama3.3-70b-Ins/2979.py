from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort the offers based on the end index
        offers.sort(key=lambda x: x[1])
        
        # Initialize a list to store the maximum profit at each index
        dp = [0] * n
        
        # Iterate over each offer
        for start, end, gold in offers:
            # If the start index is 0, the maximum profit is the gold
            if start == 0:
                dp[end] = max(dp[end], gold)
            # If the start index is not 0, the maximum profit is the maximum of the current profit and the profit from the previous index plus the gold
            else:
                dp[end] = max(dp[end], dp[start-1] + gold)
        
        # Update the dp array to store the maximum profit at each index
        for i in range(1, n):
            dp[i] = max(dp[i], dp[i-1])
        
        # Return the maximum profit at the last index
        return dp[-1]