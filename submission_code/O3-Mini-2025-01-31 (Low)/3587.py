from typing import List
import math

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # dp[day][city] represents the maximum score achievable on day "day" ending in "city"
        # Initialize dp for day 0 with starting city choices using stayScore.
        dp_prev = [stayScore[0][city] for city in range(n)]
        
        # Process each subsequent day from day 1 to day k-1.
        for day in range(1, k):
            dp_curr = [0] * n
            
            # Precompute best and second best from dp_prev, along with the corresponding index for best.
            best_val = -math.inf
            best_idx = -1
            second_val = -math.inf
            for city in range(n):
                if dp_prev[city] > best_val:
                    second_val = best_val
                    best_val = dp_prev[city]
                    best_idx = city
                elif dp_prev[city] > second_val:
                    second_val = dp_prev[city]
            
            # For each city as the destination on current day.
            for dest in range(n):
                # Option 1: Stay in the same city: add the day's staying score at that city.
                option_stay = dp_prev[dest] + stayScore[day][dest]
                
                # Option 2: Travel from some previous city:
                # We want to pick the maximum dp_prev[prev] + travelScore[prev][dest] for any valid prev.
                # To optimize this, we use the precomputed best values.
                if dest == best_idx:
                    # Cannot choose best_val because that corresponds to staying in the same city 
                    # in the travel option (if move, destination must be different than current city).
                    option_travel = second_val + travelScore[dest][dest]  # this isn't used because travelScore[dest][dest]==0
                    # But we need to iterate over all cities? Instead, we must try the possibility from the best candidate that is not dest.
                    # We'll compute option_travel by scanning dp_prev for all prev != dest.
                    option_travel = -math.inf
                    for prev in range(n):
                        if prev != dest:
                            candidate = dp_prev[prev] + travelScore[prev][dest]
                            if candidate > option_travel:
                                option_travel = candidate
                else:
                    # If dest is not best_idx, then best_val is from a different city.
                    option_travel = best_val + travelScore[best_idx][dest]
                
                dp_curr[dest] = max(option_stay, option_travel)
            
            dp_prev = dp_curr
        
        return max(dp_prev)