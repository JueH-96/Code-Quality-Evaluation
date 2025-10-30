from typing import List

class Solution:
  def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
    n = len(maxHeights)

    # dp_left[i] stores the sum of heights for a valid mountain prefix h[0...i]
    # where h[i] = maxHeights[i], h[j] <= maxHeights[j], and h[0...i] is non-decreasing.
    dp_left = [0] * n
    stack = []  # Stores (value, count) representing blocks of same height
    current_sum = 0
    for i in range(n):
      current_val_at_peak = maxHeights[i]
      # This tower itself contributes 1 to the count of towers with this height
      count_for_current_val = 1
      
      # Pop from stack while stack top block is taller or equal.
      # These taller/equal blocks must be effectively reduced to current_val_at_peak.
      while stack and stack[-1][0] >= current_val_at_peak:
        val_popped, count_popped = stack.pop()
        current_sum -= val_popped * count_popped
        count_for_current_val += count_popped # These towers join the new block
      
      # Add the new block of towers (all height current_val_at_peak)
      stack.append((current_val_at_peak, count_for_current_val))
      current_sum += current_val_at_peak * count_for_current_val
      dp_left[i] = current_sum

    # dp_right[i] stores the sum of heights for a valid mountain suffix h[i...n-1]
    # where h[i] = maxHeights[i], h[j] <= maxHeights[j], and h[i...n-1] is non-increasing.
    # Calculation is symmetric to dp_left, iterating from right to left.
    dp_right = [0] * n
    stack = []  # Reset stack
    current_sum = 0 # Reset current_sum
    for i in range(n - 1, -1, -1):
      current_val_at_peak = maxHeights[i]
      count_for_current_val = 1
      
      while stack and stack[-1][0] >= current_val_at_peak:
        val_popped, count_popped = stack.pop()
        current_sum -= val_popped * count_popped
        count_for_current_val += count_popped
        
      stack.append((current_val_at_peak, count_for_current_val))
      current_sum += current_val_at_peak * count_for_current_val
      dp_right[i] = current_sum
      
    max_total_sum = 0
    for i in range(n):
      # For each index i considered as the peak of the mountain:
      # The sum of heights in the left part (non-decreasing up to i) is dp_left[i].
      # The sum of heights in the right part (non-increasing from i) is dp_right[i].
      # maxHeights[i] (which is heights[i] at the peak) is included in both sums.
      # So, we subtract it once to get the total sum for the mountain with peak i.
      current_mountain_sum = dp_left[i] + dp_right[i] - maxHeights[i]
      if current_mountain_sum > max_total_sum:
        max_total_sum = current_mountain_sum
        
    return max_total_sum