from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        """
        Calculates the maximum possible points a tourist can earn over k days
        using dynamic programming with space optimization.
        """
        
        # dp[j] stores the max total score ending in city j on the previous day.
        # Initialize for day 0. On day 0, the tourist must start in some city j
        # and stay, so the score is stayScore[0][j].
        dp = [0] * n
        for j in range(n):
            dp[j] = stayScore[0][j]
            
        # Iterate for each subsequent day from day 1 to k-1.
        for i in range(1, k):
            # next_dp[j] will store the max total score ending in city j on day i.
            next_dp = [0] * n
            
            # For each destination city `j` on day `i`.
            for j in range(n):
                # Calculate the max score to arrive at city `j` on day `i`.
                # This is the maximum over all possible previous cities `p` from day `i-1`.
                max_score_to_j = 0 # Scores are non-negative, so 0 is a safe initial value.
                
                # Iterate over all possible previous cities `p`.
                for p in range(n):
                    score_from_previous_day = dp[p]
                    score_for_today = 0
                    
                    if p == j: # Tourist was in city `j` and stayed.
                        score_for_today = stayScore[i][j]
                    else: # Tourist was in city `p` and moved to `j`.
                        score_for_today = travelScore[p][j]
                    
                    total_score_via_p = score_from_previous_day + score_for_today

                    # Update the max score for reaching city `j` today.
                    if total_score_via_p > max_score_to_j:
                        max_score_to_j = total_score_via_p
                        
                next_dp[j] = max_score_to_j
                
            # The scores for the current day become the scores for the "previous day"
            # for the next iteration.
            dp = next_dp
            
        # After all k days, the max score is the maximum of the total scores achievable
        # in any city on the last day.
        return max(dp)