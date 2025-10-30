import math

class Solution:
  def maxScore(self, n: int, k: int, stayScore: list[list[int]], travelScore: list[list[int]]) -> int:
    # prev_dp[city_idx] stores the maximum score ending in city_idx for the previous day.
    # current_dp[city_idx] stores the maximum score ending in city_idx for the current day.
    
    # Initialize prev_dp. These will store scores at the end of day 0.
    # Using float('-inf') as an initial minimum handles edge cases gracefully.
    # The actual values stored will become integers after calculations because
    # stayScore and travelScore are integers.
    prev_dp = [float('-inf')] * n

    # --- Day 0 (day_idx = 0) ---
    # Calculate scores at the end of day 0.
    # The tourist can start in any city 's' at the beginning of day 0.
    for j in range(n):  # j is the city where the tourist is at the end of day 0
      # Option 1: Started in city j and stayed in city j for day 0.
      # Initial score before day 0 is effectively 0.
      score_if_stayed_in_j = stayScore[0][j]
      
      # Option 2: Started in city s (s != j) and moved to city j for day 0.
      max_score_if_moved_to_j = float('-inf')
      # Iterate over all possible starting cities 's'
      for s in range(n):
        if s == j: # Cannot move from j to j; this is covered by staying.
          continue
        max_score_if_moved_to_j = max(max_score_if_moved_to_j, travelScore[s][j])
      
      # prev_dp[j] is the max score to end day 0 in city j.
      # If n=1, max_score_if_moved_to_j remains float('-inf').
      # prev_dp[0] becomes max(stayScore[0][0], float('-inf')) = stayScore[0][0]. This is correct.
      prev_dp[j] = max(score_if_stayed_in_j, max_score_if_moved_to_j)

    # If k=1, the loop below (for days 1 to k-1) is skipped.
    # prev_dp already holds scores for day 0 (which is day k-1 in this case).

    # --- Days 1 to k-1 ---
    # current_dp is used to store results for the current day_idx.
    current_dp = [float('-inf')] * n 
    
    for day_idx in range(1, k):
      for j in range(n):  # j is the city where the tourist is at the end of day_idx
        # Option 1: Was in city j at end of day (day_idx-1), stayed in city j for day_idx.
        # Score = (score up to end of day_idx-1 in city j) + (score for staying in j on day_idx)
        # prev_dp[j] contains the score up to end of day_idx-1 in city j.
        # Note: prev_dp[j] will be an integer >= 1 because stayScore[0][c] >= 1.
        score_if_stayed = prev_dp[j] + stayScore[day_idx][j]
        
        # Option 2: Was in city p (p != j) at end of day (day_idx-1), moved from p to j for day_idx.
        # Score = (score up to end of day_idx-1 in city p) + (score for moving p->j on day_idx)
        max_score_if_moved = float('-inf')
        # Iterate over all possible previous cities 'p'
        for p in range(n):
          if p == j: # Cannot move from j to j if considering a "move" action.
            continue
          max_score_if_moved = max(max_score_if_moved, prev_dp[p] + travelScore[p][j])
        
        # current_dp[j] is the max score to end day_idx in city j.
        # If n=1, max_score_if_moved remains float('-inf').
        # current_dp[0] becomes max(prev_dp[0] + stayScore[day_idx][0], float('-inf')). Correct.
        current_dp[j] = max(score_if_stayed, max_score_if_moved)
      
      # Update prev_dp for the next iteration. Copy contents of current_dp to prev_dp.
      for city_idx_copy in range(n):
        prev_dp[city_idx_copy] = current_dp[city_idx_copy]

    # The result is the maximum score in any city at the end of day k-1.
    # prev_dp now holds scores for day k-1.
    # Since n >= 1, prev_dp is not empty.
    # Since stayScore items are >= 1, all values in prev_dp will be integers >= 1.
    final_max_score = float('-inf')
    for score in prev_dp:
        final_max_score = max(final_max_score, score)
            
    return int(final_max_score)