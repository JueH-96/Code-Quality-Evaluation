from typing import List
from collections import defaultdict

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # group offers by their ending index
        end_to_offers = defaultdict(list)
        for s, e, g in offers:
            end_to_offers[e].append((s, g))
        
        # dp[i] = max gold we can earn using houses in [0, i-1]
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            # default: skip selling house i-1
            best = dp[i - 1]
            
            # consider offers that end at house i-1
            for s, g in end_to_offers.get(i - 1, ()):
                best = max(best, dp[s] + g)
            
            dp[i] = best
        
        return dp[n]