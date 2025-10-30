import collections
from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # dp[i] will store the maximum gold we can earn by considering houses from 0 to i-1.
        # So, dp[0] means no houses considered (empty range), resulting in 0 gold.
        # The final answer will be dp[n], representing the maximum gold for houses 0 to n-1.
        dp = [0] * (n + 1)

        # Group offers by their ending house index.
        # offers_by_end[k] will be a list of (start, gold) tuples for all offers
        # that conclude at house index k.
        # Using a defaultdict ensures that accessing a key that doesn't exist yet
        # returns an empty list, preventing KeyError.
        offers_by_end = collections.defaultdict(list)
        for start, end, gold in offers:
            offers_by_end[end].append((start, gold))

        # Iterate through house indices from 0 to n-1.
        # In the dp array, index `i` corresponds to considering houses up to `i-1`.
        for i in range(1, n + 1):
            # Option 1: Don't take any offer that ends at house `i-1`.
            # In this scenario, the maximum profit is simply inherited from the previous state,
            # which considered houses up to `i-2`.
            dp[i] = dp[i-1]

            # Option 2: Consider taking offers that end precisely at house `i-1`.
            # We iterate through all such offers.
            # `i-1` is the actual house index being considered as an end point.
            for start_house, gold_value in offers_by_end[i-1]:
                # If we take this offer [start_house, i-1, gold_value],
                # we earn `gold_value`.
                # The houses from `start_house` to `i-1` are now occupied.
                # The maximum gold we could have earned *before* this offer starts
                # is `dp[start_house]`.
                # So, the total profit for this path would be `dp[start_house] + gold_value`.
                # We update dp[i] to be the maximum of its current value and this new potential profit.
                dp[i] = max(dp[i], dp[start_house] + gold_value)
        
        # After iterating through all possible end points (up to house n-1),
        # dp[n] will hold the maximum gold achievable for all houses from 0 to n-1.
        return dp[n]