from typing import List

class Solution:
  def minimumOperations(self, nums: List[int], target: List[int]) -> int:
    n = len(nums)
    
    # diff_i_minus_1 represents diff[i-1] in the calculation of y[i] = diff[i] - diff[i-1].
    # For i=0, diff[-1] is conceptually 0.
    diff_i_minus_1 = 0
    # total_ops will store the sum of positive y_i values.
    # Python integers handle arbitrary size, so overflow is not an issue.
    total_ops: int = 0 
    
    for i in range(n):
      # Calculate current diff value: diff[i] = target[i] - nums[i]
      diff_i = target[i] - nums[i]
      
      # Calculate y_i = diff[i] - diff[i-1]
      # This y_i is an element of the "difference of differences" array.
      y_i_val = diff_i - diff_i_minus_1
      
      if y_i_val > 0:
        # If y_i_val is positive, it contributes to the sum of operations.
        # This corresponds to operations starting or increasing in intensity.
        total_ops += y_i_val
      
      # Update diff_i_minus_1 for the next iteration. It becomes the "previous" diff value.
      diff_i_minus_1 = diff_i
      
    # After the loop, diff_i_minus_1 stores diff[n-1] (the last element of the diff array).
    # The final element of the conceptual y_array is y_n = diff[n] - diff[n-1].
    # diff[n] is conceptually 0 (target state for an element beyond the array).
    # So, y_n = 0 - diff[n-1].
    y_n_val = 0 - diff_i_minus_1 # This is effectively -diff[n-1]
    
    if y_n_val > 0:
      # This case occurs if diff[n-1] was negative.
      # A positive y_n_val means that diff[n-1] (which was negative) is "lower" than diff[n] (which is zero).
      # This change also contributes to the sum of positive y_values.
      total_ops += y_n_val
      
    return total_ops