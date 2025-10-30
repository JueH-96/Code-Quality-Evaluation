from typing import List

class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        """
        Computes the maximum absolute difference between any two adjacent
        elements in the circular list `nums`. Adjacency is considered in a
        circular sense, so the first and last elements are also adjacent.

        Time Complexity: O(n), where n = len(nums)
        Space Complexity: O(1)
        """
        n = len(nums)
        # At least two elements per constraints; handle generically
        max_diff = 0
        for i in range(n):
            diff = abs(nums[i] - nums[(i + 1) % n])  # adjacent in circular sense
            if diff > max_diff:
                max_diff = diff
        return max_diff