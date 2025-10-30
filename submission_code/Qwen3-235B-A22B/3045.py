from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)
        for k in range(n):
            rotated = nums[-k:] + nums[:-k]
            if rotated == sorted_nums:
                return k
        return -1