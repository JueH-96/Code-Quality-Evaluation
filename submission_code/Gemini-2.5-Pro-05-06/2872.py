from typing import List

class Solution:
  def maxArrayValue(self, nums: List[int]) -> int:
    n = len(nums)
    
    # Constraints: 1 <= nums.length <= 10^5.
    # This means n is always at least 1.
    
    # If there's only one element, that element itself is the maximum possible value,
    # as no operations can be performed.
    if n == 1:
      return nums[0]
      
    # 'current_sum' holds the sum of a contiguous block of elements,
    # accumulated by applying operations from right to left.
    # Initially, it's just the last element of the array.
    current_sum = nums[n-1]
    
    # 'max_overall_val' tracks the maximum value encountered. This can be any
    # 'current_sum' formed during the process, or any individual element `nums[i]`
    # that starts a new sum sequence if it cannot be merged.
    # Initialized with nums[n-1] because all numbers are positive (>=1),
    # so the last element is a valid candidate for the maximum and is the
    # first value `current_sum` takes.
    max_overall_val = nums[n-1]
    
    # Iterate from the second-to-last element (index n-2) down to the first element (index 0).
    for i in range(n - 2, -1, -1):
      # 'val_i' is the current element nums[i] we are considering.
      val_i = nums[i]
      
      # The operation is: if nums[i] <= nums[i+1], replace nums[i+1] with nums[i]+nums[i+1]
      # and delete nums[i]. The sum nums[i]+nums[i+1] effectively takes the position of nums[i].
      # In our right-to-left scan, `current_sum` plays the role of this `nums[i+1]`
      # (which might itself be an accumulated sum from further right).
      # We check if `val_i` (our `nums[i]`) can merge with `current_sum`.
      if val_i <= current_sum:
        # If `val_i` is less than or equal to `current_sum`, we can perform the operation.
        # `val_i` is merged into `current_sum`.
        # This new sum is conceptually at index `i`.
        current_sum = val_i + current_sum
      else:
        # If `val_i` is greater than `current_sum`, we cannot merge `val_i` with `current_sum`.
        # The chain of accumulation is broken. The `current_sum` we had
        # (representing sum from `nums[i+1]` to some `nums[k]`) is a finalized sum from that block.
        # `val_i` now starts a new potential accumulation sequence. So, `current_sum` is reset to `val_i`.
        current_sum = val_i
      
      # After `current_sum` is updated (either by merging or by resetting),
      # this new `current_sum` value is a candidate for the overall maximum element
      # that can be formed in the array. We update `max_overall_val` if this `current_sum` is larger.
      max_overall_val = max(max_overall_val, current_sum)
      
    return max_overall_val