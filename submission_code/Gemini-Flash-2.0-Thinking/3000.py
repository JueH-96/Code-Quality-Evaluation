from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_abs_diff = float('inf')

        for i in range(n):
            for j in range(n):
                if abs(i - j) >= x:
                    abs_diff = abs(nums[i] - nums[j])
                    min_abs_diff = min(min_abs_diff, abs_diff)

        return min_abs_diff