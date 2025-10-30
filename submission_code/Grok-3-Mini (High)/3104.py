import bisect
from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)
        num_set = set(nums)
        count = 0
        for s in range(0, n + 1):
            if s not in num_set:
                cnt_less_s = bisect.bisect_left(sorted_nums, s)
                if cnt_less_s == s:
                    count += 1
        return count