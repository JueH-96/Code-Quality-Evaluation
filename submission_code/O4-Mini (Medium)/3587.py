from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # dp_prev[c] will hold the max points we can have ending in city c after the previous day
        dp_prev = [0] * n
        
        # Iterate over each day
        for day in range(k):
            dp_curr = [0] * n
            for dest in range(n):
                # Option 1: stay in the same city 'dest'
                best = dp_prev[dest] + stayScore[day][dest]
                # Option 2: travel from any other city 'src' to 'dest'
                for src in range(n):
                    if src != dest:
                        best = max(best, dp_prev[src] + travelScore[src][dest])
                dp_curr[dest] = best
            dp_prev = dp_curr
        
        # The answer is the best we can do on the last day in any city
        return max(dp_prev)