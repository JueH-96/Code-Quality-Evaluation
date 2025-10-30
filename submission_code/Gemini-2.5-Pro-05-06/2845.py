import math
from typing import List

class Solution:
  def findValueOfPartition(self, nums: List[int]) -> int:
    # Sort the array. The logic explained above shows that the minimum
    # partition value will be the minimum difference between two adjacent
    # elements in the sorted array.
    nums.sort()
    
    min_difference = math.inf
    
    # Iterate through the sorted array to find the minimum adjacent difference.
    # nums.length is at least 2, so this loop runs at least once.
    for i in range(len(nums) - 1):
      difference = nums[i+1] - nums[i]
      
      # Since the array is sorted, nums[i+1] - nums[i] is always non-negative.
      if difference < min_difference:
        min_difference = difference
        
        # Optimization: If a difference of 0 is found, it's the smallest possible
        # value for |max(nums1) - min(nums2)|, so we can return immediately.
        if min_difference == 0:
          return 0
            
    # The problem constraints ensure nums contains integers, so differences are integers.
    # min_difference will hold an integer value.
    return int(min_difference)