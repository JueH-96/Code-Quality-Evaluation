from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        if k == 0:
            return 0
        
        # Initialize previous day's DP values for day 0
        prev_dp = [0] * n
        for j in range(n):
            max_travel = 0
            for m in range(n):
                if m != j:
                    if travelScore[m][j] > max_travel:
                        max_travel = travelScore[m][j]
            prev_dp[j] = max(stayScore[0][j], max_travel)
        
        # Iterate over days from 1 to k-1
        for i in range(1, k):
            current_dp = [0] * n
            for j in range(n):
                # Option 1: Stay in city j
                stay_val = prev_dp[j] + stayScore[i][j]
                
                # Option 2: Move to city j from another city
                max_move = 0
                for m in range(n):
                    if m != j:
                        current_val = prev_dp[m] + travelScore[m][j]
                        if current_val > max_move:
                            max_move = current_val
                current_dp[j] = max(stay_val, max_move)
            
            prev_dp = current_dp.copy()
        
        return max(prev_dp)