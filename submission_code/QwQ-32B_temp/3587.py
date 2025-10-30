from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize previous DP array for day 0
        prev_dp = [0] * n
        for j in range(n):
            max_travel = max(travelScore[m][j] for m in range(n))
            prev_dp[j] = max(stayScore[0][j], max_travel)
        
        # Iterate through each subsequent day
        for i in range(1, k):
            current_dp = [0] * n
            for j in range(n):
                # Option 1: stay in current city j
                option1 = prev_dp[j] + stayScore[i][j]
                
                # Option 2: come from any city m to j
                max_option2 = -float('inf')
                for m in range(n):
                    current_val = prev_dp[m] + travelScore[m][j]
                    if current_val > max_option2:
                        max_option2 = current_val
                option2 = max_option2
                
                # Choose the better option
                current_dp[j] = max(option1, option2)
            prev_dp = current_dp
        
        return max(prev_dp)