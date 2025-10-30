from typing import List

class Solution:
  def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
    n = len(nums)
    max_length = 0

    for l in range(n):
      # nums[l] is the potential start of a subarray.
      # Condition 1: nums[l] must be even.
      # Condition 3: All elements, including nums[l], must be <= threshold.
      if nums[l] % 2 == 0 and nums[l] <= threshold:
        # If nums[l] satisfies these, it forms a valid subarray of length 1: [nums[l]].
        # current_length tracks the length of the valid subarray nums[l...r] found so far,
        # starting with nums[l...l] which has length 1.
        current_length = 1
        max_length = max(max_length, current_length)
        
        # Try to extend this subarray to the right with elements nums[l+1], nums[l+2], ...
        # r is the index of the element being considered to add to the subarray.
        for r in range(l + 1, n):
          # The element to potentially add is nums[r].
          # The previous element in the sequence was nums[r-1].
          
          # Condition 3 for nums[r]: must be <= threshold.
          if nums[r] > threshold:
            # nums[r] violates threshold, so subarray cannot be extended further.
            break 
          
          # Condition 2: Parity of nums[r] must be different from parity of nums[r-1].
          if nums[r] % 2 == nums[r-1] % 2:
            # Parities are the same, so alternating property is violated.
            # Subarray cannot be extended further.
            break
          
          # If both conditions pass, nums[r] successfully extends the subarray.
          current_length += 1
          max_length = max(max_length, current_length)
      # else: nums[l] is odd or nums[l] > threshold.
      # It cannot start a valid alternating subarray. So, we move to the next l.
      
    return max_length