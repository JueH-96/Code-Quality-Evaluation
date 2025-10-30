from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # dp[i] will store the maximum profit considering houses from 0 to i-1.
        # The array size is n+1 because dp[n] will store the max profit for houses 0 to n-1.
        # dp[0] = 0, representing no houses considered yet, profit is 0.
        dp = [0] * (n + 1)

        # Group offers by their end house.
        # offers_by_end_house[j] will be a list of [start, gold] for offers ending at house j.
        # House indices are 0 to n-1.
        # Create a list of lists, where the index corresponds to the end house index.
        offers_by_end_house = [[] for _ in range(n)]
        for start, end, gold in offers:
            offers_by_end_house[end].append([start, gold])

        # Iterate through index i from 1 to n.
        # dp[i] represents the maximum profit considering houses 0 to i-1.
        # To calculate dp[i], we consider decisions related to house i-1.
        for i in range(1, n + 1):
            # Option 1: Don't take any offer that ends exactly at house i-1.
            # In this case, the maximum profit up to house i-1 is the same as the
            # maximum profit up to house i-2, which is dp[i-1].
            dp[i] = dp[i-1]

            # Option 2: Consider taking an offer that ends exactly at house i-1.
            # House index is i-1. We check offers ending at this house index.
            current_house_index = i - 1
            # offers_by_end_house[current_house_index] contains offers ending at house i-1.
            # Iterate through all offers that end at current_house_index.
            for start, gold in offers_by_end_house[current_house_index]:
                # If we take this offer [start, current_house_index, gold],
                # the houses from 'start' to 'current_house_index' are covered.
                # No other chosen offer can overlap with this range.
                # The maximum profit we could have accumulated from houses before 'start'
                # (i.e., houses 0 to start-1) is given by dp[start].
                # The total profit would be dp[start] + gold.
                # We take the maximum profit achievable by considering all offers ending at
                # current_house_index and comparing with the profit from Option 1.
                dp[i] = max(dp[i], dp[start] + gold)

        # After iterating through all i up to n, dp[n] holds the maximum profit
        # considering houses 0 to n-1, which is the total range.
        return dp[n]