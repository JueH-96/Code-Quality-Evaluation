import math
from typing import List

class Solution:
  def maxFrequencyScore(self, nums: List[int], k: int) -> int:
    n = len(nums)
    nums.sort()

    prefix_sum = [0] * (n + 1)
    for i in range(n):
      prefix_sum[i+1] = prefix_sum[i] + nums[i]

    # Returns the sum of nums[start_idx...end_idx]
    # using 0-based indexing for the original sorted nums array.
    # nums is 0-indexed. prefix_sum[j] = sum(nums[0]...nums[j-1]).
    # So sum(nums[a...b]) = prefix_sum[b+1] - prefix_sum[a].
    def get_sum_range(start_idx: int, end_idx: int) -> int:
      if start_idx > end_idx: # Empty range
        return 0
      return prefix_sum[end_idx+1] - prefix_sum[start_idx]

    # Checks if it's possible to make f elements equal with at most k cost.
    def can(f: int) -> bool:
      # A frequency of 1 is always possible with 0 cost by picking any element
      # and not changing it. The general logic below correctly computes 0 cost for f=1.
      
      # Iterate through all windows of size f. A window is nums[i ... i+f-1].
      for i in range(n - f + 1):
        # Determine the median element of the current window nums[i...i+f-1].
        # The window has 0-indexed elements w_0, ..., w_{f-1}.
        # The median is w_((f-1)//2). (This is the left median if f is even).
        # Its index in the window is median_idx_in_window.
        # Its index in the sorted nums array is median_actual_idx.
        median_idx_in_window = (f - 1) // 2
        median_actual_idx = i + median_idx_in_window
        target_val = nums[median_actual_idx] # This is the value all f elements will be changed to.

        current_cost = 0
        
        # Cost for elements to the left of target_val in the window.
        # These are nums[i ... median_actual_idx-1].
        # There are median_idx_in_window such elements.
        num_left_elements = median_idx_in_window 
        sum_left_vals = get_sum_range(i, median_actual_idx - 1)
        # Each nums[j] for j < median_actual_idx becomes target_val. Cost is target_val - nums[j].
        # Total cost for left part: sum(target_val - nums[j]) = target_val * num_left_elements - sum_left_vals.
        current_cost += (target_val * num_left_elements) - sum_left_vals
        
        # Cost for elements to the right of target_val in the window.
        # These are nums[median_actual_idx+1 ... i+f-1].
        # There are (f-1) - median_idx_in_window such elements.
        num_right_elements = (f - 1) - median_idx_in_window
        sum_right_vals = get_sum_range(median_actual_idx + 1, i + f - 1)
        # Each nums[j] for j > median_actual_idx becomes target_val. Cost is nums[j] - target_val.
        # Total cost for right part: sum(nums[j] - target_val) = sum_right_vals - target_val * num_right_elements.
        current_cost += sum_right_vals - (target_val * num_right_elements)
        
        if current_cost <= k:
          return True # Found a window of f elements that can be made equal with cost <= k.
          
      return False # No window of size f can be made equal with cost <= k.

    # Binary search for the maximum frequency f.
    # Smallest possible score is 1 (pick any element, cost 0), as n >= 1 from constraints.
    ans = 1 
    low = 1 # Min possible frequency
    high = n # Max possible frequency
    
    while low <= high:
      mid_f = low + (high - low) // 2
      # mid_f will always be >= 1 because low starts at 1.
      # So, no need to check `if mid_f == 0`.
      
      if can(mid_f):
        ans = mid_f         # mid_f is achievable, try for a possibly larger frequency.
        low = mid_f + 1 
      else:
        high = mid_f - 1    # mid_f is not achievable, try a smaller frequency.
        
    return ans