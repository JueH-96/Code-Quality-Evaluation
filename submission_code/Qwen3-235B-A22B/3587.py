from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize previous DP array with the best possible scores after day 0
        prev_dp = [0] * n
        for c in range(n):
            max_val = float('-inf')
            for d in range(n):
                if d == c:
                    candidate = stayScore[0][d]
                else:
                    candidate = travelScore[d][c]
                if candidate > max_val:
                    max_val = candidate
            prev_dp[c] = max_val
        
        # Process each subsequent day
        for day in range(1, k):
            curr_dp = [0] * n
            for c in range(n):
                # Calculate the points if staying in city c
                stay_points = prev_dp[c] + stayScore[day][c]
                
                # Calculate the maximum points from moving to city c from any other city
                max_move = float('-inf')
                for prev_c in range(n):
                    if prev_c != c:
                        candidate = prev_dp[prev_c] + travelScore[prev_c][c]
                        if candidate > max_move:
                            max_move = candidate
                
                # Determine the best option between staying and moving
                curr_dp[c] = max(stay_points, max_move)
            
            # Update previous DP array to current for the next iteration
            prev_dp = curr_dp
        
        # The result is the maximum value in the DP array after all days are processed
        return max(prev_dp)