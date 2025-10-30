from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        for k in range(n):
            shifted = nums[-k:] + nums[:-k]
            if shifted == sorted(nums):
                return k
        return -1