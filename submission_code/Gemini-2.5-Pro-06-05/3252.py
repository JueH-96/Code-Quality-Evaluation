from typing import List

class Solution:
  def incremovableSubarrayCount(self, nums: List[int]) -> int:
    n = len(nums)
    count = 0

    # Helper function to check if an array is strictly increasing
    def is_strictly_increasing(arr: List[int]) -> bool:
      for i in range(len(arr) - 1):
        if arr[i] >= arr[i + 1]:
          return False
      return True

    # Iterate through all possible start and end indices of the subarray to remove
    for i in range(n):
      for j in range(i, n):
        # The subarray to remove is nums[i:j+1]
        # Construct the array with the subarray removed
        remaining_array = nums[:i] + nums[j+1:]
        
        # Check if the resulting array is strictly increasing
        if is_strictly_increasing(remaining_array):
          count += 1
    
    return count