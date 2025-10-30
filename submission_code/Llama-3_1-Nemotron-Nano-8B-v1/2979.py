import bisect
from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort the offers by their end time
        sorted_offers = sorted(offers, key=lambda x: x[1])
        m = len(sorted_offers)
        if m == 0:
            return 0
        
        # Extract end times for binary search
        end_times = [offer[1] for offer in sorted_offers]
        
        # Initialize DP array
        dp = [0] * m
        dp[0] = max(0, sorted_offers[0][2])
        
        for i in range(1, m):
            current_gold = sorted_offers[i][2]
            start_i = sorted_offers[i][0]
            
            # Find the latest non-overlapping offer using binary search
            j = bisect.bisect_left(end_times, start_i, 0, i) - 1
            if j >= 0:
                current_gold += dp[j]
            
            # Determine the maximum profit up to the i-th offer
            dp[i] = max(dp[i-1], current_gold)
        
        return dp[-1]