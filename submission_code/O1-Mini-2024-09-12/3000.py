from typing import List
import bisect

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        s = []
        min_diff = float('inf')
        for i in range(len(nums)):
            if i >= x:
                bisect.insort(s, nums[i - x])
            if s:
                pos = bisect.bisect_left(s, nums[i])
                if pos < len(s):
                    min_diff = min(min_diff, abs(nums[i] - s[pos]))
                if pos > 0:
                    min_diff = min(min_diff, abs(nums[i] - s[pos - 1]))
        return min_diff