from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize previous day's dp
        prev_dp = [0] * n
        for c in range(n):
            option1 = stayScore[0][c]
            option2 = max(travelScore[p][c] for p in range(n))
            prev_dp[c] = max(option1, option2)
        
        # Iterate through each day from 1 to k-1
        for d in range(1, k):
            curr_dp = [0] * n
            for c in range(n):
                # Option 1: Stay in city c
                option1 = prev_dp[c] + stayScore[d][c]
                # Option 2: Travel to city c from any city p
                option2 = max(prev_dp[p] + travelScore[p][c] for p in range(n))
                # Choose the maximum of the two options
                curr_dp[c] = max(option1, option2)
            # Update previous dp for the next iteration
            prev_dp = curr_dp
        
        # The answer is the maximum points over all cities on the last day
        return max(prev_dp)