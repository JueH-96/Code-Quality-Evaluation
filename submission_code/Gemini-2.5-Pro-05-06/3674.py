from typing import List

class Solution:
  def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
    n = len(nums)
    count = 0
    
    for i in range(n):
      current_cost = 0
      # b_val_at_prev_idx stores the value of the last element in the
      # modified non-decreasing version of nums[i...j-1].
      # For the first element nums[i], its modified value is nums[i].
      b_val_at_prev_idx = nums[i] 
      
      for j in range(i, n):
        if j == i:
          # Subarray is nums[i:i+1] = [nums[i]]
          # Cost is 0.
          # b_val_at_curr_idx for nums[i] is nums[i].
          # current_cost is already 0.
          # b_val_at_prev_idx is already nums[i] (initialized before inner loop).
          pass
        else:
          # Subarray is nums[i:j+1], considering extending from nums[i...j-1] with nums[j].
          # b_val_at_prev_idx is the b-value of nums[j-1] in the modified nums[i...j-1].
          # We need to make nums[j] >= b_val_at_prev_idx.
          # The cost to do this for nums[j] is max(0, b_val_at_prev_idx - nums[j]).
          added_cost = 0
          if b_val_at_prev_idx > nums[j]:
            added_cost = b_val_at_prev_idx - nums[j]
          
          if current_cost + added_cost > k:
            # If adding nums[j] makes cost exceed k, no further extension from nums[i...]
            # will be valid, as costs are non-decreasing.
            break
          
          current_cost += added_cost
          
          # Update b_val_at_prev_idx to be the b-value for nums[j] in modified nums[i...j].
          # This will be max(nums[j], b_val_at_prev_idx).
          if nums[j] > b_val_at_prev_idx:
            b_val_at_prev_idx = nums[j]
            
        # If current_cost <= k, this subarray nums[i...j] is valid.
        count += 1
        
    return count