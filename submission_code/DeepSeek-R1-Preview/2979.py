import bisect
from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort the offers by their end values
        offers.sort(key=lambda x: x[1])
        # dp_list stores tuples of (end, max_profit) and starts with a base case
        dp_list = [(-1, 0)]
        # ends is a list to keep track of end values for binary search
        ends = [-1]
        
        for offer in offers:
            start, end, gold = offer
            target = start - 1
            # Find the rightmost end value <= target using binary search
            idx = bisect.bisect_right(ends, target) - 1
            if idx >= 0:
                prev_profit = dp_list[idx][1]
            else:
                prev_profit = 0
            current_profit = prev_profit + gold
            # If the current profit is higher than the last recorded profit, update the dp_list
            if current_profit > dp_list[-1][1]:
                dp_list.append((end, current_profit))
                ends.append(end)
        
        # The maximum profit is the last value in dp_list
        return dp_list[-1][1]