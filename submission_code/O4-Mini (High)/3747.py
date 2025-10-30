from typing import List

class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        """
        Computes the maximum absolute difference between adjacent elements in a circular array.
        """
        max_diff = 0
        n = len(nums)
        for i in range(n):
            # Next index wraps around using modulo for circular behavior
            j = (i + 1) % n
            diff = abs(nums[i] - nums[j])
            if diff > max_diff:
                max_diff = diff
        return max_diff