from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # dp[i][j] will be the maximum score achievable after day i if the tourist is in city j.
        # We use a 1-D dp array that we update for each day.
        
        # For day 0, the tourist can choose any starting city.
        # Two options on day 0:
        #   1. Start in city j and stay: score = stayScore[0][j]
        #   2. Start in some city u (u != j) and travel to city j on day 0: score = travelScore[u][j]
        # (For n==1, only option is to stay because there's no other city to travel from.)
        dp = [0] * n
        if n == 1:
            dp[0] = stayScore[0][0]
        else:
            for j in range(n):
                # Option 1: Stay at city j.
                score_stay = stayScore[0][j]
                # Option 2: Travel from some other city u to city j.
                best_travel = 0  # since travelScore values are non-negative.
                for u in range(n):
                    if u == j:
                        continue
                    if travelScore[u][j] > best_travel:
                        best_travel = travelScore[u][j]
                dp[j] = max(score_stay, best_travel)
                
        # Process days 1 to k-1.
        for day in range(1, k):
            new_dp = [0] * n
            for j in range(n):
                # Option 1: Stay in city j.
                stay_option = dp[j] + stayScore[day][j]
                # Option 2: Travel to city j from some other city u.
                travel_option = float('-inf')
                for u in range(n):
                    if u == j:
                        continue  # must move from a different city.
                    candidate = dp[u] + travelScore[u][j]
                    if candidate > travel_option:
                        travel_option = candidate
                new_dp[j] = max(stay_option, travel_option)
            dp = new_dp
        
        # The answer is the maximum points obtainable on the final day over all cities.
        return max(dp)