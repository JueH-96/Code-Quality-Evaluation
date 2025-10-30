from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        """
        dp[day][city] – maximum points that can be collected after finishing `day`
        and being in `city`.
        We keep only the previous day row (dp_prev) because every day depends only
        on the results of the preceding one.
        """
        
        # after the imaginary day -1 we may choose any city to start from → score 0
        dp_prev = [0] * n
        
        for day in range(k):
            dp_curr = [-10 ** 9] * n        # very small sentinel value
            for curr in range(n):
                base = dp_prev[curr]
                
                # option 1: stay where we are
                stay_val = base + stayScore[day][curr]
                if stay_val > dp_curr[curr]:
                    dp_curr[curr] = stay_val
                
                # option 2: move to a different city
                row = travelScore[curr]
                for dest in range(n):
                    if dest == curr:
                        continue            # moving to the same city brings 0, staying is always as good or better
                    move_val = base + row[dest]
                    if move_val > dp_curr[dest]:
                        dp_curr[dest] = move_val
                        
            dp_prev = dp_curr               # proceed to the next day
        
        return max(dp_prev)                 # best score obtainable after k days