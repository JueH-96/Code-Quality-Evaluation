import collections
from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        """
        Calculates the maximum profit by accepting a set of non-overlapping offers for houses
        using dynamic programming.
        """
        
        # dp[i] stores the maximum profit considering houses 0 to i-1.
        dp = [0] * (n + 1)
        
        # Group offers by their end point for efficient lookup.
        # The map stores: end_house -> list of (start_house, gold)
        offers_by_end = collections.defaultdict(list)
        for start, end, gold in offers:
            offers_by_end[end].append((start, gold))
            
        # Iterate from i = 1 to n, where i represents the number of houses considered.
        # In each iteration, we compute the max profit for houses 0..i-1.
        for i in range(1, n + 1):
            # The current house being considered as a potential endpoint is `i-1`.
            end_house = i - 1

            # Option 1: The max profit is at least the profit from the previous state (dp[i-1]).
            # This covers the case where house `end_house` is not sold as an endpoint.
            dp[i] = dp[i-1]
            
            # Option 2: An offer ends at `end_house`.
            # We check if taking one of these offers gives a better profit.
            if end_house in offers_by_end:
                for start, gold in offers_by_end[end_house]:
                    # The total profit would be the offer's gold plus the max profit
                    # achievable before this offer's range, which is dp[start].
                    dp[i] = max(dp[i], dp[start] + gold)
                    
        return dp[n]