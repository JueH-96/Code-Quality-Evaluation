import bisect
from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort the offers by their end time
        offers_sorted = sorted(offers, key=lambda x: x[1])
        m = len(offers_sorted)
        if m == 0:
            return 0
        
        # Precompute the ends array for binary search
        ends = [offer[1] for offer in offers_sorted]
        
        # Initialize DP array
        dp = [0] * m
        dp[0] = offers_sorted[0][2]
        
        for i in range(1, m):
            s_i = offers_sorted[i][0]
            target = s_i - 1
            
            # Find the largest index j where ends[j] <= target in the first i elements
            j = bisect.bisect_right(ends, target, 0, i) - 1
            prev = dp[j] if j >= 0 else 0
            take = offers_sorted[i][2] + prev
            not_take = dp[i-1]
            
            dp[i] = max(take, not_take)
        
        return dp[-1]