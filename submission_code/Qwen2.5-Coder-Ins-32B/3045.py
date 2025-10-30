from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)
        
        # Find the index where the sorted array starts in the original array
        for i in range(n):
            if nums[i:] + nums[:i] == sorted_nums:
                return i
        
        return -1