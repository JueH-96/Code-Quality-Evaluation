import bisect
from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        count = 0
        for k in range(n + 1):
            m = bisect.bisect_left(nums, k)
            if m != k:
                continue
            if m < n and nums[m] <= k:
                continue
            count += 1
        return count