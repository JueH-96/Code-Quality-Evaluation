import math
from typing import List

class Solution:
  """
  Finds the maximum score achievable by traversing the array `nums` starting from index 0,
  following specific movement rules and scoring penalties.
  Uses dynamic programming with O(N) time complexity and O(1) space complexity.
  """
  def maxScore(self, nums: List[int], x: int) -> int:
    """
    Calculates the maximum score achievable by visiting elements in `nums`.

    Args:
      nums: A 0-indexed list of integers representing the values at each position.
      x: A positive integer representing the penalty incurred when moving between
         positions with values of different parities.

    Returns:
      The maximum total score achievable according to the rules.
    """
    n = len(nums)
    
    # Initialize variables to track the maximum score ending with an even number
    # and the maximum score ending with an odd number.
    # Using a very large negative integer effectively represents negative infinity,
    # ensuring that initial calculations involving non-existent paths of a certain parity
    # do not incorrectly influence the maximum score.
    # The value should be smaller than any possible achievable score.
    # Minimum possible score can be roughly nums[0] - (n-1)*x. With max N=10^5, max x=10^6,
    # min score could be around 1 - 10^11. Using -10^18 or larger negative number is safe.
    NEG_INF = -2 * 10**18 
    
    max_even = NEG_INF  # Tracks the maximum score of any valid path ending with an even number
    max_odd = NEG_INF   # Tracks the maximum score of any valid path ending with an odd number
    
    # Initialize the state based on the first element nums[0].
    # Any valid path must start at index 0. The score at this point is nums[0].
    if nums[0] % 2 == 0:
        # If the first number is even, the maximum score ending in even is nums[0].
        max_even = nums[0]
    else:
        # If the first number is odd, the maximum score ending in odd is nums[0].
        max_odd = nums[0]
        
    # Iterate through the array starting from the second element (index 1).
    for i in range(1, n):
        num_i = nums[i]
        parity_i = num_i % 2 # Determine the parity of the current number nums[i].
        
        if parity_i == 0: # Current number nums[i] is even
            # Calculate the maximum possible score for a path ending specifically at index i.
            # A path ending at i must come from some index j < i.
            # If the path comes from j where nums[j] is even (same parity), the score is `max_even + nums[i]`.
            # If the path comes from j where nums[j] is odd (different parity), the score is `max_odd + nums[i] - x`.
            # We take the maximum of these two possibilities. `max_even` and `max_odd` store the maximum scores
            # ending in even/odd parity respectively *before* considering index i.
            # The calculation `max(max_even, max_odd - x)` finds the best score achievable just before landing on index i,
            # considering potential penalties. Adding num_i gives the score ending at i.
            
            # Note: The simplified calculation `num_i + max(max_even, max_odd - x)` works correctly even if one
            # of max_even or max_odd is NEG_INF. The `max` function will correctly select the valid score path,
            # or the less negative path if both are essentially NEG_INF.
            current_score = num_i + max(max_even, max_odd - x)
            
            # Update max_even. It should hold the maximum score found so far for *any* path ending in an even number,
            # considering all indices up to i. This could be the previous max_even, or the new score `current_score` we just calculated.
            max_even = max(max_even, current_score)

        else: # Current number nums[i] is odd
            # Similar logic applies if nums[i] is odd.
            # Calculate the maximum score ending at index i.
            # Option 1: Path comes from an odd number (same parity). Score is `max_odd + nums[i]`.
            # Option 2: Path comes from an even number (different parity). Score is `max_even + nums[i] - x`.
            current_score = num_i + max(max_odd, max_even - x)

            # Update max_odd. It tracks the maximum score found so far for *any* path ending in an odd number up to index i.
            max_odd = max(max_odd, current_score)

    # After iterating through all elements, max_even and max_odd hold the maximum scores
    # achievable for paths ending in even and odd numbers, respectively, across all possible ending positions.
    # The overall maximum score achievable is the maximum of these two values.
    # Since we start at index 0 and nums[0] >= 1, at least one path exists (visiting only index 0).
    # Thus, at least one of max_even or max_odd will hold a valid score (not NEG_INF).
    # The max() function correctly returns the highest valid score.
    return max(max_even, max_odd)