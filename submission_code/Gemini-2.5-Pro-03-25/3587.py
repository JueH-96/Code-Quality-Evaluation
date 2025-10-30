from typing import List

class Solution:
  """
  Calculates the maximum possible score a tourist can earn over k days
  visiting n cities with given stay and travel scores using dynamic programming.
  The time complexity is O(k * n^2) and space complexity is O(n).
  """
  def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
    """
    Args:
      n: The number of cities (1 <= n <= 200).
      k: The number of days (1 <= k <= 200).
      stayScore: A 2D list (k x n) where stayScore[i][j] is the score for staying
                 in city j on day i (1 <= stayScore[i][j] <= 100).
      travelScore: A 2D list (n x n) where travelScore[i][j] is the score for traveling
                   from city i to city j (0 <= travelScore[i][j] <= 100, travelScore[i][i] == 0).
    
    Returns:
      The maximum possible score the tourist can earn after k days.
    """

    # current_dp[j] stores the maximum score achievable ending in city j
    # after the previous day's actions. Initialized to 0 for the state before day 0,
    # reflecting that the tourist can start in any city with an initial score of 0.
    # The size of this list is n.
    current_dp = [0] * n
    
    # Iterate through each day of the journey, from day 0 to day k-1.
    # The loop runs k times.
    for day in range(k):
        # next_dp[j] will store the maximum score achievable ending in city j
        # after the actions of the current day (`day`).
        # Initialize with -1. Since all scores involved are non-negative, 
        # any actual path will result in a score >= 0. Using -1 ensures that 
        # the first calculated score for each city correctly becomes the initial maximum for that state.
        next_dp = [-1] * n 
        
        # Iterate through all possible cities `curr` the tourist could be in at the START of the current day.
        # This inner loop runs n times.
        for curr in range(n):
            # Retrieve the maximum score to reach city `curr` at the start of this day.
            score_at_start_of_day = current_dp[curr]
            
            # Since initial scores are 0 and subsequent scores added are non-negative, 
            # score_at_start_of_day will always be >= 0 for any state reachable from the start.
            # A check like `if score_at_start_of_day == -1: continue` is not strictly needed here
            # but could be useful in problems where reaching certain states might be impossible.

            # Option 1: Stay in the current city `curr`.
            # Calculate the score obtained by staying: current score + score for staying today.
            stay_option_score = score_at_start_of_day + stayScore[day][curr]
            # Update the maximum score for ending the day in city `curr`. We take the maximum
            # because there might be multiple ways to arrive at city `curr` at the end of the day
            # (e.g., staying in `curr`, or moving from another city to `curr`). We want the path
            # that yields the highest score.
            next_dp[curr] = max(next_dp[curr], stay_option_score)
            
            # Option 2: Move from city `curr` to another city `dest`.
            # Iterate through all possible destination cities `dest`.
            # This innermost loop runs n times.
            for dest in range(n):
                # The problem specifies moving to *another* city.
                if dest != curr:
                    # Calculate the score obtained by moving: current score + score for traveling.
                    move_option_score = score_at_start_of_day + travelScore[curr][dest]
                    # Update the maximum score for ending the day in city `dest`. Similar to the stay
                    # option, we take the maximum score found so far for ending in city `dest`.
                    next_dp[dest] = max(next_dp[dest], move_option_score)

        # After evaluating all options (stay or move) for all possible starting cities `curr` for the current day,
        # `next_dp` holds the maximum scores for ending in each city after day `day`'s actions.
        # Update `current_dp` to `next_dp` to prepare for the next day's calculation. This effectively
        # rolls the state forward one day.
        current_dp = next_dp
        
    # After the loop completes (k days processed), `current_dp` contains the maximum scores
    # achievable ending in each city `j` after exactly k days.
    # The final result is the maximum score among all possible ending cities, as the tourist
    # could end their journey in any city.
    
    # If k = 0, the loop is skipped, `current_dp` remains `[0]*n`, and `max(current_dp)` correctly returns 0.
    # The `max` function correctly finds the maximum value in the list `current_dp`.
    # Since n >= 1, the list is never empty.
    # Since scores are non-negative and initial state is 0, max score will be >= 0.
    return max(current_dp)