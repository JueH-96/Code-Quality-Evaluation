import math
from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # dp[d][c] stores the maximum score achievable by the end of day 'd',
        # with the tourist located in city 'c'.
        # The dp table will have 'k' rows (for days 0 to k-1) and 'n' columns (for cities 0 to n-1).
        dp = [[0] * n for _ in range(k)]

        # Base case: Day 0
        # On day 0, the tourist chooses a starting city and stays there.
        # The score is simply the stayScore for that city on day 0.
        for city in range(n):
            dp[0][city] = stayScore[0][city]

        # Fill the DP table for subsequent days (from day 1 to k-1)
        for day in range(1, k):
            for current_city in range(n):
                # Option 1: Stay in the current_city
                # The score is the maximum score from the previous day ending in current_city
                # plus the stayScore for the current day in current_city.
                score_if_stay = dp[day - 1][current_city] + stayScore[day][current_city]

                # Option 2: Move from another city to the current_city
                # Initialize with negative infinity to ensure any valid travel score is greater.
                score_if_travel = -math.inf 
                
                # Iterate through all possible previous cities from which the tourist could have moved.
                for prev_city in range(n):
                    # According to the problem description, "Move to another city" implies prev_city != current_city.
                    # Also, travelScore[i][i] is 0, so staying would typically be worse with travelScore than stayScore.
                    if prev_city != current_city:
                        score_if_travel = max(score_if_travel, dp[day - 1][prev_city] + travelScore[prev_city][current_city])

                # The maximum score for `current_city` on the `current day` is the maximum
                # of either staying in the city or traveling to it from another city.
                dp[day][current_city] = max(score_if_stay, score_if_travel)
        
        # After filling the DP table for all days, the maximum possible points
        # will be the maximum score achievable on the last day (day k-1) across all cities.
        max_total_score = max(dp[k - 1])
        
        return max_total_score