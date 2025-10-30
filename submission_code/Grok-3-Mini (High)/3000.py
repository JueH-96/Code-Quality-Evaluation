import bisect
from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        if x == 0:
            return 0
        min_diff = float('inf')
        sorted_values = []
        for j in range(x, len(nums)):
            add_val = nums[j - x]
            bisect.insort(sorted_values, add_val)
            val = nums[j]
            pos = bisect.bisect_left(sorted_values, val)
            current_diff = float('inf')
            if pos < len(sorted_values):
                current_diff = min(current_diff, abs(val - sorted_values[pos]))
            if pos - 1 >= 0:
                current_diff = min(current_diff, abs(val - sorted_values[pos - 1]))
            min_diff = min(min_diff, current_diff)
        return int(min_diff)