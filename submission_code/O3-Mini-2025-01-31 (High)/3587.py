from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Special case: only one city => can only stay
        if n == 1:
            return sum(day_score[0] for day_score in stayScore)
        
        # dp[j] will represent the maximum score obtainable starting from city j at the beginning of the next day.
        dp = [0] * n  # Base case: after k days, no more score
        
        # Process days in reverse order: day k-1 down to day 0.
        for day in range(k - 1, -1, -1):
            new_dp = [0] * n
            for city in range(n):
                # Option 1: Stay on the current day at the same city.
                stay_option = stayScore[day][city] + dp[city]
                
                # Option 2: Travel to a different city.
                travel_option = -10**9  # Start with a very low value.
                for dest in range(n):
                    if dest == city:
                        continue  # Can't travel to the same city.
                    candidate = travelScore[city][dest] + dp[dest]
                    if candidate > travel_option:
                        travel_option = candidate
                
                # Choose the best option for this city at the current day.
                new_dp[city] = stay_option if stay_option > travel_option else travel_option
            dp = new_dp  # Update dp for next iteration (previous day)
        
        # The tourist can start at any city, so return the maximum across all starting positions.
        return max(dp)