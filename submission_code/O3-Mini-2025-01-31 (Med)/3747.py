from typing import List

class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        n = len(nums)
        max_diff = 0
        for i in range(n):
            # next index in circular way
            next_index = (i + 1) % n
            diff = abs(nums[i] - nums[next_index])
            if diff > max_diff:
                max_diff = diff
        return max_diff