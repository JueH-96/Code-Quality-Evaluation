from typing import List
from collections import defaultdict

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Create a dictionary to store offers ending at each house
        end_offers = defaultdict(list)
        for start, end, gold in offers:
            end_offers[end].append((start, gold))
        
        # Initialize dp array where dp[i] is the maximum gold we can earn up to house i
        dp = [0] * (n + 1)
        
        # Fill the dp array
        for i in range(1, n + 1):
            # Start with the maximum gold we can earn without selling any house ending at i
            dp[i] = dp[i - 1]
            # Check all offers ending at house i
            for start, gold in end_offers[i - 1]:
                dp[i] = max(dp[i], dp[start] + gold)
        
        return dp[n]