from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Initialize the dp array with -1
        dp = [-1] * (n + 1)
        dp[0] = 0
        
        # Sort the offers by the end point
        offers.sort(key=lambda x: x[1])
        
        # Iterate over the offers
        for start, end, gold in offers:
            # Iterate from the start to the end
            for i in range(start, end + 1):
                # If the dp value at i is not -1, update the dp value at i
                if dp[i] != -1:
                    dp[i] = max(dp[i], dp[start] + gold)
        
        # Return the maximum value in the dp array
        return max(dp)