import bisect
from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort the offers based on the end index of each offer
        offers.sort(key=lambda x: x[1])
        m = len(offers)
        # Extract the end positions of each offer for binary search
        ends = [offer[1] for offer in offers]
        # Initialize the dp array where dp[i] represents the maximum profit considering the first i offers
        dp = [0] * (m + 1)
        
        for i in range(1, m + 1):
            current_offer = offers[i - 1]
            start, end, gold = current_offer
            # Target is the last house position before the current offer's start
            target = start - 1
            # Find the rightmost offer which ends <= target using bisect_right
            k = bisect.bisect_right(ends, target, 0, i)
            # Calculate the profit if current offer is taken
            current_profit = dp[k] + gold
            # Update dp[i] to be the maximum profit either by taking or skipping the current offer
            dp[i] = max(dp[i - 1], current_profit)
        
        # The last element of dp contains the maximum profit considering all offers
        return dp[m]