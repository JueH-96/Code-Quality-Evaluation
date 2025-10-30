import math
from typing import List

class Solution:
  def maximumTotalCost(self, nums: List[int]) -> int:
    n = len(nums)
    # Constraints state n >= 1.
    # If n=1, the loop for i in range(1,n) will not run.
    # The result will be max(dp_add_val, dp_sub_val) initialized for i=0.
    # dp_add_val_for_0 = nums[0], dp_sub_val_for_0 = -math.inf. max(nums[0], -math.inf) = nums[0].
    # This is correct for n=1. E.g. input [0], output 0. input [5], output 5.

    # dp_add_val stores the maximum total cost of nums[0...current_index]
    # where nums[current_index] contributes positively to the sum.
    # This means nums[current_index] is an even-indexed element within its own subarray.
    # (0-indexed from the start of that subarray, e.g. nums[k] or ...-nums[k+j-1]+nums[k+j] where j is even).
    dp_add_val = nums[0]
    
    # dp_sub_val stores the maximum total cost of nums[0...current_index]
    # where nums[current_index] contributes negatively to the sum.
    # This means nums[current_index] is an odd-indexed element within its own subarray.
    # (e.g. ...+nums[k+j-1]-nums[k+j] where j is odd).
    # For nums[0], it must be the first element of its subarray, so it cannot be subtracted.
    dp_sub_val = -math.inf

    # Iterate from the second element (index i=1) up to n-1
    for i in range(1, n):
      current_num = nums[i]
      
      # Store DP values from the previous iteration (for index i-1)
      prev_dp_add = dp_add_val
      prev_dp_sub = dp_sub_val

      # Calculate dp_add_val for current_num (nums[0...i] ending with +current_num)
      # Scenario 1: current_num starts a new subarray [current_num].
      #   Cost of nums[0...i-1] is max(prev_dp_add, prev_dp_sub).
      #   Total cost = max(prev_dp_add, prev_dp_sub) + current_num.
      # Scenario 2: current_num extends the previous subarray, and current_num is added
      #   (e.g., subarray ends ... -nums[i-1] + current_num).
      #   This implies nums[i-1] was subtracted. Cost of nums[0...i-1] is prev_dp_sub.
      #   Total cost = prev_dp_sub + current_num.
      # dp_add_val is the maximum of these two scenarios.
      # max( (max(prev_dp_add, prev_dp_sub) + current_num), (prev_dp_sub + current_num) )
      # This simplifies to: max(prev_dp_add, prev_dp_sub) + current_num.
      dp_add_val = max(prev_dp_add, prev_dp_sub) + current_num
      
      # Calculate dp_sub_val for current_num (nums[0...i] ending with -current_num)
      # current_num must extend a previous subarray where nums[i-1] was added
      # (e.g., subarray ends ... +nums[i-1] - current_num).
      # Cost of nums[0...i-1] (ending with +nums[i-1]) is prev_dp_add.
      # Total cost = prev_dp_add - current_num.
      dp_sub_val = prev_dp_add - current_num
      
    # The final answer is the maximum possible cost for the whole array nums[0...n-1],
    # considering both possibilities for how nums[n-1] contributes to the sum.
    return max(dp_add_val, dp_sub_val)