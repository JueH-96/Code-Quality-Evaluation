from typing import List
import bisect

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        offers.sort(key=lambda x: x[1])  # Sort offers by end time
        m = len(offers)
        if m == 0:
            return 0
        ends = [offer[1] for offer in offers]  # List of end times
        dp = [0] * (m + 1)
        for i in range(1, m + 1):
            start_val = offers[i - 1][0]  # Start time of current offer
            # Binary search to find the largest index k where ends[k] < start_val
            idx = bisect.bisect_left(ends, start_val)
            p_idx = idx - 1
            # Value if we take the current offer
            take_value = offers[i - 1][2] + dp[p_idx + 1]
            # Value if we don't take the current offer
            not_take = dp[i - 1]
            # Maximum of the two options
            dp[i] = max(not_take, take_value)
        return dp[m]