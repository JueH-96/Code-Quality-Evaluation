import bisect
from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        count = 0
        for k in range(n + 1):
            left = bisect.bisect_left(nums, k)
            right = bisect.bisect_right(nums, k)
            if (right - left) != 0:
                continue
            if left == k:
                count += 1
        return count