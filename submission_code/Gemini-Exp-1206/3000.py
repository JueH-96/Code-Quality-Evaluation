from typing import List
import bisect

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        if x == 0:
            return 0
        ans = float('inf')
        sorted_nums = []
        for i in range(x, n):
            bisect.insort(sorted_nums, nums[i - x])
            idx = bisect.bisect_left(sorted_nums, nums[i])
            if idx > 0:
                ans = min(ans, abs(nums[i] - sorted_nums[idx - 1]))
            if idx < len(sorted_nums):
                ans = min(ans, abs(nums[i] - sorted_nums[idx]))
        return ans