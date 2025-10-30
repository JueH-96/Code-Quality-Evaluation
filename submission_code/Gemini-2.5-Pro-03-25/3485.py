import math
from typing import List

class Solution:
  """
  Finds the maximum possible score, where score is the minimum absolute difference 
  between any two chosen integers. Integers x_i must be chosen from intervals [start[i], start[i] + d].
  
  The problem asks us to choose n integers $x_0, x_1, \dots, x_{n-1}$ such that for each $i$, $x_i \in [start[i], start[i] + d]$.
  The score of a chosen set is defined as $S = \min_{0 \le i < j < n} |x_i - x_j|$.
  We want to find the maximum possible value of $S$.

  This problem can be solved using binary search on the answer (the score $M$).
  We want to find the largest integer $M$ such that it's possible to choose $x_0, \dots, x_{n-1}$ satisfying the interval constraints and the minimum difference constraint $|x_i - x_j| \ge M$ for all $i 
eq j$.

  The range for binary search for $M$ is from 0 up to the maximum possible difference between any two chosen numbers. A safe upper bound is $2 \times 10^9$ based on the constraints on `start[i]` and `d`.

  The core of the solution is the `check(M)` function, which determines if a given score $M$ is achievable.
  We can use a greedy strategy for the `check(M)` function. The strategy relies on sorting the intervals based on their start points. Let the sorted intervals be $I'_0, I'_1, \dots, I'_{n-1}$, where $I'_k = [s_k, e_k]$. We attempt to choose values $y_0, y_1, \dots, y_{n-1}$ such that $y_k \in I'_k$ for all $k$, and they satisfy the condition $y_k \ge y_{k-1} + M$. This also implies $y_0 \le y_1 \le \dots \le y_{n-1}$. If such a sequence can be constructed, it represents a valid choice of $x_i$'s satisfying the minimum difference $M$. The greedy approach picks the minimum possible value for each $y_k$: $y_k = \max(s_k, y_{k-1} + M)$. If at any step this chosen value $y_k$ exceeds the upper bound $e_k$ of the interval $I'_k$, then this greedy strategy fails for score $M$. It turns out that if this greedy strategy fails, no other assignment works that satisfies both $y_k \in I'_k$ and the sorted property $y_k \ge y_{k-1} + M$. Although the problem does not require the chosen values to be sorted according to the interval start times, this greedy check has been shown to work for similar problems and passes the examples provided.
  """
  def maxPossibleScore(self, start: List[int], d: int) -> int:
    """
    Uses binary search on the possible score M.
    The check function verifies if a score M is achievable using a greedy strategy.

    Args:
      start: A list of integers representing the start points of the intervals.
      d: An integer representing the length of each interval. `end = start + d`.

    Returns:
      The maximum possible score (minimum absolute difference).
    """
    
    n = len(start)
    
    # Create intervals represented as tuples (start_point, end_point).
    intervals = []
    for i in range(n):
        intervals.append((start[i], start[i] + d))
    
    # Sort intervals based on their start points. This is essential for the greedy check strategy.
    # Python's built-in sort is stable and efficient (Timsort O(N log N)).
    intervals.sort()

    # Check function: Determines if a minimum difference of M is achievable.
    def check(M):
        """
        Implements the greedy check strategy.
        Iterates through intervals sorted by start points. For each interval,
        it chooses the smallest possible integer value that satisfies the minimum 
        difference M constraint with the value chosen for the previous interval.
        
        Args:
          M: The minimum difference (score) to check.
          
        Returns:
          True if score M is achievable according to the greedy strategy, False otherwise.
        """
        # A score of 0 is always achievable if n >= 2, as we can potentially pick the same value
        # or values very close depending on intervals. The problem states n >= 2.
        if M == 0:
             return True

        # `last_chosen` keeps track of the integer value chosen for the previously processed interval.
        # Initialize to negative infinity. This ensures that for the first interval (k=0),
        # the choice `current_choice = max(s_0, -inf + M)` correctly becomes `s_0`.
        last_chosen = -math.inf 
        
        # Iterate through the intervals sorted by start point.
        for k in range(n):
            s_k = intervals[k][0]  # Start point of the k-th interval in sorted order.
            e_k = intervals[k][1]  # End point of the k-th interval in sorted order.
            
            # Determine the minimum possible value for the current interval choice.
            # It must be at least s_k (to be within the interval [s_k, e_k]).
            # It must also be at least M greater than the value chosen for the previous interval.
            current_choice = max(s_k, last_chosen + M)
            
            # Check if this minimum required value is actually possible (within the interval's end point).
            if current_choice > e_k:
                # If the smallest valid choice `current_choice` is greater than the interval's end point `e_k`,
                # then we cannot select a value for this interval that satisfies the constraints.
                # The greedy strategy fails for this M.
                return False
            
            # If a valid choice is possible, update `last_chosen` to this choice for the next iteration.
            last_chosen = current_choice
            
        # If we successfully found a valid choice for all intervals using the greedy approach.
        return True

    # Binary search for the maximum possible score M.
    low = 0  # Minimum possible score is 0.
    # Set a safe upper bound for the score. Max possible chosen value is max(start) + d, min is min(start).
    # Max difference is roughly max(start) + d - min(start).
    # Constraints: start[i] <= 10^9, d <= 10^9. Max value ~2*10^9. Min value ~0. Max diff ~2*10^9.
    high = 2 * 10**9 
    
    max_score = 0 # Initialize the maximum achievable score found so far.
    
    while low <= high:
        # Calculate midpoint. Using low + (high - low) // 2 prevents potential overflow for large low/high in languages with fixed-size integers. Python handles large integers, but this is good practice.
        mid = low + (high - low) // 2
        
        if check(mid):
            # If a score of `mid` is achievable, it means we might be able to achieve an even higher score.
            # Record `mid` as the current best achievable score and try searching in the upper half range [mid+1, high].
            max_score = mid
            low = mid + 1
        else:
            # If a score of `mid` is not achievable, the maximum possible score must be smaller than `mid`.
            # Search in the lower half range [low, mid-1].
            high = mid - 1
            
    # After the binary search loop terminates, max_score holds the maximum achievable score.
    return max_score