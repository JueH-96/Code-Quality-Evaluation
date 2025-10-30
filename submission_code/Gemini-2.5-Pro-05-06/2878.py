from typing import List

class Solution:
  def checkArray(self, nums: List[int], k: int) -> bool:
    n = len(nums)
    
    # operations_started_at[j] stores the number of operations
    # that were started at index j. This is decided greedily.
    # We use an array of size n. This is fine given N <= 10^5.
    operations_started_at = [0] * n 
    
    # current_carried_decrease is the sum of operations_started_at[p]
    # for all p such that an operation started at p affects the current element nums[i].
    # These are operations started in the window [max(0, i-k+1), i-1].
    current_carried_decrease = 0
    
    for i in range(n):
      # If i >= k, the operation that started at index i-k no longer affects nums[i].
      # An operation started at i-k covers indices [i-k, i-k+k-1] = [i-k, i-1].
      # So, remove its contribution from current_carried_decrease.
      if i >= k:
        current_carried_decrease -= operations_started_at[i-k]
      
      # effective_val_i is the value of nums[i] after considering all
      # operations that started at an index p < i and cover index i.
      # Note: nums[i] is the original value. We don't modify the input nums array.
      effective_val_i = nums[i] - current_carried_decrease
      
      # If effective_val_i is negative, it means previous operations
      # (chosen to zero out earlier elements nums[0]...nums[i-1])
      # have already made nums[i] negative. This is impossible if we only aim for 0,
      # as operations only decrease elements.
      if effective_val_i < 0:
        return False
      
      # If effective_val_i is 0, nums[i] is already effectively zero.
      # No new operation needs to start at index i.
      # operations_started_at[i] is already 0 from initialization.
      if effective_val_i == 0:
        pass # No operation needed starting at i.
      # If effective_val_i is positive, we must start effective_val_i operations
      # at index i to make nums[i] effectively zero. This is the greedy choice.
      elif effective_val_i > 0:
        # Check if we *can* start an operation of length k at index i.
        # An operation on subarray nums[i...i+k-1] is valid if i+k-1 < n,
        # which is equivalent to i <= n-k.
        # If i > n-k, we cannot start an operation at i that stays within array bounds.
        # If nums[i] is still positive (i.e. effective_val_i > 0) at this point,
        # and we cannot start an operation, then it's impossible to make it zero.
        if i > n - k: # Check if current index i is too far to the right
          return False
        
        operations_started_at[i] = effective_val_i
      
      # Add the effect of operations started at i (if any) to current_carried_decrease.
      # This amount will affect subsequent elements nums[i+1], ..., nums[i+k-1].
      current_carried_decrease += operations_started_at[i]
      
    # If the loop completes without returning False, it means we have found
    # a valid sequence of operations according to the greedy strategy.
    return True