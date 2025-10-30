from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize dp for day 0
        prev_dp = []
        for c in range(n):
            stay = stayScore[0][c]
            travel_max = max(travelScore[s][c] for s in range(n))
            prev_dp.append(max(stay, travel_max))
        
        # Iterate through days 1 to k-1
        for i in range(1, k):
            curr_dp = []
            for c in range(n):
                option1 = stayScore[i][c] + prev_dp[c]
                option2 = max(travelScore[p][c] + prev_dp[p] for p in range(n))
                curr_dp.append(max(option1, option2))
            prev_dp = curr_dp
        
        return max(prev_dp)