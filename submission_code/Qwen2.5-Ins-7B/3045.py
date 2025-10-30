from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        if nums == sorted(nums):
            return 0
        
        count = 0
        for i in range(n - 1):
            if nums[i] > nums[i + 1]:
                count += 1
                start = i + 1
            if count > 1:
                return -1
        
        if count == 1 and nums[0] >= nums[-1]:
            return n - start
        return -1