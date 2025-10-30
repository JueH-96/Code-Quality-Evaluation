import bisect
import math

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        min_diff = float('inf')
        sorted_prefix = []
        for i in range(len(nums)):
            if i >= x:
                bisect.insort(sorted_prefix, nums[i - x])
                idx = bisect.bisect_left(sorted_prefix, nums[i])
                if idx < len(sorted_prefix):
                    min_diff = min(min_diff, abs(nums[i] - sorted_prefix[idx]))
                if idx > 0:
                    min_diff = min(min_diff, abs(nums[i] - sorted_prefix[idx - 1]))
        return min_diff