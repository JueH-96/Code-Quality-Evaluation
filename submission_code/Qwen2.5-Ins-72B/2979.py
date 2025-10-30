from typing import List
from bisect import bisect_left

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        offers.sort(key=lambda x: x[1])
        dp = [0] * (len(offers) + 1)
        
        for i, (start, end, gold) in enumerate(offers, 1):
            j = bisect_left(offers, [start, float('inf'), 0])
            dp[i] = max(dp[i - 1], dp[j] + gold)
        
        return dp[-1]