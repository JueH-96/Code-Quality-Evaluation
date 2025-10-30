from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_diff = float('inf')
        sorted_nums = sorted(nums[i] for i in range(n) if i + x < n)
        for i in range(x, n):
            idx = bisect.bisect_left(sorted_nums, nums[i])
            if idx < len(sorted_nums):
                min_diff = min(min_diff, abs(nums[i] - sorted_nums[idx]))
            if idx > 0:
                min_diff = min(min_diff, abs(nums[i] - sorted_nums[idx-1]))
        return min_diff