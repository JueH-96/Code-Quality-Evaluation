from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # dp_prev[j] = max score up to previous day ending in city j
        # Day 0: must start and stay in some city j
        dp_prev = stayScore[0].copy()
        # Precompute columns of travelScore for faster access: travelScore_T[j][u] = travelScore[u][j]
        travelScore_T = list(zip(*travelScore))
        
        # Process days 1 through k-1
        for day in range(1, k):
            dp_curr = [0] * n
            stay_today = stayScore[day]
            for j in range(n):
                # Option 1: stay in city j
                best = dp_prev[j] + stay_today[j]
                # Option 2: travel from any city u to j, earn travelScore[u][j]
                col_j = travelScore_T[j]
                max_travel = 0
                # find max over u of dp_prev[u] + travelScore[u][j]
                for u, t in enumerate(col_j):
                    val = dp_prev[u] + t
                    if val > max_travel:
                        max_travel = val
                # pick the better of stay vs travel
                if max_travel > best:
                    best = max_travel
                dp_curr[j] = best
            dp_prev = dp_curr
        
        # The answer is the max achievable on the last day across all cities
        return max(dp_prev)