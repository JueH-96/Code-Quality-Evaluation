import bisect
from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ans = 0
        for k in range(n + 1):
            left = bisect.bisect_left(nums, k)
            right = bisect.bisect_right(nums, k)
            cnt_less = left
            cnt_eq = right - left
            if cnt_less == k and cnt_eq == 0:
                ans += 1
        return ans