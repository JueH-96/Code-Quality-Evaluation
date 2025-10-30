from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # dp[d][j] will be the max score on day d if we end in city j.
        # We can roll arrays to save space.
        
        # Initialize for day 0: start in any city j and stay there
        prev = [stayScore[0][j] for j in range(n)]
        
        # Process days 1 through k-1
        for day in range(1, k):
            curr = [-10**18] * n
            for j in range(n):
                # Option 1: we were in j yesterday and stay today
                curr[j] = prev[j] + stayScore[day][j]
                # Option 2: we travel from some other city p to j
                #          and earn travelScore[p][j]
                best_travel = 0
                for p in range(n):
                    if p == j:
                        continue
                    val = prev[p] + travelScore[p][j]
                    if val > curr[j]:
                        curr[j] = val
                # curr[j] now holds the best ending at j on this day
            prev = curr
        
        # The answer is the maximum over all cities on the last day
        return max(prev)