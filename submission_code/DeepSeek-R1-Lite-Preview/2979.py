from typing import List
import bisect

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort the offers based on end time
        offers.sort(key=lambda x: x[1])
        m = len(offers)
        end_times = [offer[1] for offer in offers]
        # Initialize DP array
        dp = [0] * (m + 1)
        
        for k in range(1, m + 1):
            offer = offers[k - 1]
            start_i = offer[0]
            gold_i = offer[2]
            # Find the last offer that ends before start_i
            index = bisect.bisect_left(end_times, start_i)
            j = index - 1
            if j >= 0:
                dp[k] = max(dp[k - 1], gold_i + dp[j + 1])
            else:
                dp[k] = max(dp[k - 1], gold_i)
        
        return dp[m]