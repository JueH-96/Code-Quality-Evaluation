import math
from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # min_prefix[i] stores the minimum value in nums[0...i]
        # This is used to find the minimum value to the left of nums[j]
        min_prefix = [0] * n
        min_prefix[0] = nums[0]
        for i in range(1, n):
            min_prefix[i] = min(min_prefix[i-1], nums[i])
            
        # min_suffix[i] stores the minimum value in nums[i...n-1]
        # This is used to find the minimum value to the right of nums[j]
        min_suffix = [0] * n
        min_suffix[n-1] = nums[n-1]
        for i in range(n - 2, -1, -1): # Iterate backwards from n-2 down to 0
            min_suffix[i] = min(min_suffix[i+1], nums[i])
            
        min_overall_sum = math.inf # Initialize with positive infinity to find the minimum sum
        
        # Iterate through all possible middle elements (j) of a mountain triplet.
        # 'j' must be at least 1 (to have an element nums[i] to its left).
        # 'j' must be at most n-2 (to have an element nums[k] to its right).
        for j in range(1, n - 1):
            # min_left_val is the smallest value found at an index i < j.
            # This value comes from min_prefix[j-1], which stores min(nums[0]...nums[j-1]).
            min_left_val = min_prefix[j-1]
            
            # min_right_val is the smallest value found at an index k > j.
            # This value comes from min_suffix[j+1], which stores min(nums[j+1]...nums[n-1]).
            min_right_val = min_suffix[j+1]
            
            # Check the mountain conditions:
            # nums[i] < nums[j] (represented by min_left_val < nums[j])
            # nums[k] < nums[j] (represented by min_right_val < nums[j])
            if min_left_val < nums[j] and min_right_val < nums[j]:
                current_sum = min_left_val + nums[j] + min_right_val
                min_overall_sum = min(min_overall_sum, current_sum)
                
        # If min_overall_sum is still infinity, it means no valid mountain triplet was found.
        if min_overall_sum == math.inf:
            return -1
        else:
            # Otherwise, return the minimum sum found.
            # The sum will already be an integer since it's a sum of integers.
            return int(min_overall_sum)