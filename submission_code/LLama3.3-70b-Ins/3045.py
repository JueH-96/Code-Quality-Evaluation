from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        min_val = min(nums)
        min_index = nums.index(min_val)
        
        # Check if the array is already sorted
        if nums == sorted(nums):
            return 0
        
        # Check if the array can be sorted by right shifts
        if nums[min_index:] + nums[:min_index] == sorted(nums):
            return min_index
        else:
            return -1