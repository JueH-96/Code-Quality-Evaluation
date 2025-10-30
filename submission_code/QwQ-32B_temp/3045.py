from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        # Check if already sorted
        if all(nums[i] <= nums[i+1] for i in range(n-1)):
            return 0
        
        # Find all breakpoints where nums[i] > nums[i+1]
        breakpoints = []
        for i in range(n - 1):
            if nums[i] > nums[i+1]:
                breakpoints.append(i)
        
        # There must be exactly one breakpoint
        if len(breakpoints) != 1:
            return -1
        
        i = breakpoints[0]
        
        # Check if the last element is <= first element
        if nums[-1] > nums[0]:
            return -1
        
        # Calculate required shifts: the length of the second part
        return n - (i + 1)