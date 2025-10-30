import bisect
from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        offers.sort(key=lambda x: x[1])
        dp = [(-1, 0)]
        max_profit = 0

        for start, end, gold in offers:
            # Find the latest offer that doesn't conflict
            pos = bisect.bisect_right(dp, (start - 1, float('inf')))
            if pos == 0:
                profit = gold
            else:
                profit = dp[pos - 1][1] + gold
            max_profit = max(max_profit, profit)
            if profit > dp[-1][1]:
                dp.append((end, profit))
        
        return max_profit