import bisect
from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        max_beauty = 0
        for i in range(n):
            target = nums[i]
            lower_bound = target - k
            upper_bound = target + k
            left_index = bisect.bisect_left(nums, lower_bound)
            right_index = bisect.bisect_right(nums, upper_bound)
            count = right_index - left_index
            max_beauty = max(max_beauty, count)
        return max_beauty