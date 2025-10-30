from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        sorted_nums = sorted(nums)
        n = len(nums)
        for k in range(n):
            shifted = nums[-k:] + nums[:-k]
            if shifted == sorted_nums:
                return k
        return -1