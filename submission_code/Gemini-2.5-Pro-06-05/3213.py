from typing import List

class Solution:
  def countSubarrays(self, nums: List[int], k: int) -> int:
    """
    Calculates the number of subarrays where the maximum element of the array
    appears at least k times using a sliding window approach.
    """
    
    # According to constraints, nums is non-empty.
    max_val = max(nums)
    
    ans = 0
    left = 0
    max_val_count = 0
    
    # Iterate through the array with the right pointer of the window.
    for right, num in enumerate(nums):
      # If the current element is the maximum value, increment its count in the window.
      if num == max_val:
        max_val_count += 1
      
      # While the window is valid (contains at least k max_val's),
      # shrink it from the left to find the boundary.
      while max_val_count >= k:
        # If the element at the left pointer is a max_val, decrement the count
        # as it's about to be excluded from the window.
        if nums[left] == max_val:
          max_val_count -= 1
        left += 1
      
      # After the while loop, `left` is the smallest index such that
      # nums[left...right] has fewer than k max_vals. This means any subarray
      # ending at `right` and starting at an index from 0 to `left - 1` is valid.
      # The number of such starting positions is `left`.
      ans += left
      
    return ans