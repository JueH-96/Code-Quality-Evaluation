import bisect
from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort the offers based on their end times
        offers.sort(key=lambda x: x[1])
        m = len(offers)
        if m == 0:
            return 0
        ends = [offer[1] for offer in offers]
        dp = [0] * m
        dp[0] = offers[0][2]
        
        for i in range(1, m):
            start_i = offers[i][0]
            target = start_i - 1
            # Find the rightmost offer that ends before the current start
            j = bisect.bisect_right(ends, target) - 1
            take = offers[i][2]
            if j >= 0:
                take += dp[j]
            # The current maximum is the max of taking or not taking the current offer
            dp[i] = max(dp[i-1], take)
        
        return dp[-1]