from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize previous day's maximum scores
        prev_dp = [0] * n
        for j in range(n):
            prev_dp[j] = stayScore[0][j]
        
        # Iterate through each day starting from day 1 (second day)
        for i in range(1, k):
            curr_dp = [0] * n
            for j in range(n):
                # Calculate the option of staying in the current city
                stay_option = prev_dp[j] + stayScore[i][j]
                
                # Calculate the maximum travel option from any other city to current city j
                max_travel = 0
                for c in range(n):
                    if c == j:
                        continue
                    current_val = prev_dp[c] + travelScore[c][j]
                    if current_val > max_travel:
                        max_travel = current_val
                
                # Determine the maximum between staying and traveling
                curr_dp[j] = max(stay_option, max_travel)
            
            # Update previous day's scores for next iteration
            prev_dp = curr_dp
        
        # The maximum score after k days is the maximum of the last day's scores
        return max(prev_dp)