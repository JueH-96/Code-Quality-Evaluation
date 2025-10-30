from typing import List
import bisect

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort offers based on the end of the range
        # Each offer is: [start, end, gold]
        offers.sort(key=lambda x: x[1])
        
        # dp array: dp[i] is a tuple (end_index, maximum_gold) among offers processed so far
        # We will use binary search on dp endpoints; initially, before any offer, "end" = -1, profit = 0
        dp = [(-1, 0)]
        
        for s, e, gold in offers:
            # Binary search for the last offer in dp with end < s (non-overlap condition)
            # We want the largest dp[j][0] such that dp[j][0] < s
            # We can use bisect_right on dp based on the tuple (s-1, big number)
            # However, since dp sorted by first component, we want to find index using bisect_right
            # then subtract 1 to get valid offer.
            idx = bisect.bisect_right(dp, (s - 1, float('inf')))
            if idx > 0:
                candidate_profit = dp[idx - 1][1] + gold
            else:
                candidate_profit = gold
            
            # If candidate profit is better than the last dp profit, update dp.
            # We want to maintain dp as sorted list of (end, profit) with increasing profits.
            if candidate_profit > dp[-1][1]:
                dp.append((e, candidate_profit))
        
        return dp[-1][1]