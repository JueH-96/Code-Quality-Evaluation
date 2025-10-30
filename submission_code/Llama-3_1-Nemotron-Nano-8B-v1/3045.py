from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        sorted_nums = sorted(nums)
        if nums == sorted_nums:
            return 0
        
        count = 0
        n = len(nums)
        for i in range(n - 1):
            if nums[i] > nums[i + 1]:
                count += 1
                if count > 1:
                    return -1
        
        if nums[-1] > nums[0]:
            return -1
        
        pos = sorted_nums.index(nums[0])
        return pos