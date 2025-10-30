from typing import List

class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        max_diff = 0
        n = len(nums)
        # Compute differences between consecutive elements
        for i in range(1, n):
            diff = abs(nums[i] - nums[i-1])
            if diff > max_diff:
                max_diff = diff
        # Check the circular pair: last and first
        circular_diff = abs(nums[-1] - nums[0])
        if circular_diff > max_diff:
            max_diff = circular_diff
        return max_diff