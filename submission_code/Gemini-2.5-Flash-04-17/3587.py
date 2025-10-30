import sys
from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # dp[i][j] = max score ending in city j after day i
        # We use space optimization, only storing results from the previous day.
        # dp_prev[j] = max score ending in city j after day i-1

        # Base case: Day 0 (i=0)
        # dp_curr[j] will store max score ending in city j after day 0
        # Initialize with a very small number as we are taking maximums.
        # Since scores are non-negative (stay >= 1, travel >= 0),
        # -sys.maxsize is a safe initial minimum.
        dp_curr = [-sys.maxsize] * n

        # On day 0, the tourist chooses a starting city `start_city` and makes a move.
        # We consider all possible starting cities `start_city`.
        # From `start_city`, they can either stay or travel.
        for start_city in range(n):
            # Option 1: Start in `start_city` and stay on day 0. Ends in `start_city`.
            # Score is stayScore[0][start_city].
            # The max score ending in `start_city` after day 0 is at least this.
            dp_curr[start_city] = max(dp_curr[start_city], stayScore[0][start_city])

            # Option 2: Start in `start_city` and travel to `dest` on day 0. Ends in `dest`.
            # Score is travelScore[start_city][dest].
            # The max score ending in `dest` after day 0 is at least this (if dest != start_city).
            for dest in range(n):
                if dest != start_city:
                    dp_curr[dest] = max(dp_curr[dest], travelScore[start_city][dest])

        # After processing all possible starting cities and their day 0 moves,
        # dp_curr now holds the max score ending in each city after day 0.
        # This becomes the base for day 1 calculation.
        dp_prev = dp_curr

        # Transitions: Day 1 to k-1
        # For each day i (from 1 up to k-1)
        for i in range(1, k):
            # dp_curr[j] will store max score ending in city j after day i
            # Initialize with a very small number
            dp_curr = [-sys.maxsize] * n

            # To end in city `j` after day `i`, the tourist must have ended in city `prev` after day `i-1`
            # and then made a move on day `i` to reach `j`.
            # We iterate through all possible previous cities `prev` (where they were at the end of day i-1).
            for prev in range(n):
                # Get the maximum score achieved up to the end of day i-1, ending in city `prev`.
                score_ending_prev_day = dp_prev[prev]

                # If score_ending_prev_day is -sys.maxsize, it means city `prev` was not reachable
                # by day i-1 with any valid score. No moves from `prev` are possible that improve the score.
                # However, adding a non-negative score to -sys.maxsize still results in a very small number,
                # which will be correctly handled by max(). So, the check isn't strictly required.

                # From city `prev`, on day `i`, the tourist can make a move:
                # Option 1: Stay in `prev`. Ends in `prev`.
                # Score for staying is stayScore[i][prev].
                # Total score ending in `prev` after staying = score_ending_prev_day + stayScore[i][prev].
                # Update the maximum score ending in city `prev` after day `i`.
                dp_curr[prev] = max(dp_curr[prev], score_ending_prev_day + stayScore[i][prev])

                # Option 2: Travel from `prev` to `dest`. Ends in `dest`.
                # Score for traveling is travelScore[prev][dest].
                # Total score ending in `dest` after traveling = score_ending_prev_day + travelScore[prev][dest].
                # Update the maximum score ending in city `dest` after day `i`.
                for dest in range(n):
                    if dest != prev: # Must travel to a different city
                        dp_curr[dest] = max(dp_curr[dest], score_ending_prev_day + travelScore[prev][dest])

            # After computing max scores for all cities after day i,
            # these results become the basis for the next day's calculation.
            dp_prev = dp_curr

        # After completing all k days (from 0 to k-1), the maximum score
        # is the maximum value among all possible ending cities on day k-1.
        # This is the maximum value in the final dp_prev array.
        return max(dp_prev)