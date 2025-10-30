import bisect
from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort the offers based on their end points
        offers.sort(key=lambda x: x[1])
        m = len(offers)
        if m == 0:
            return 0
        
        ends = [offer[1] for offer in offers]
        starts = [offer[0] for offer in offers]
        golds = [offer[2] for offer in offers]
        
        # dp[i] represents the maximum profit considering the first i offers (0..i-1)
        dp = [0] * (m + 1)
        
        for i in range(m):
            start_i = starts[i]
            gold_i = golds[i]
            
            # Find the largest index j where ends[j] < start_i using binary search
            insertion_point = bisect.bisect_left(ends, start_i, 0, i)
            j = insertion_point - 1
            
            if j >= 0:
                current_profit = gold_i + dp[j + 1]
            else:
                current_profit = gold_i
            
            # Update dp[i+1] to be the maximum of taking or not taking the current offer
            dp[i + 1] = max(dp[i], current_profit)
        
        return dp[m]