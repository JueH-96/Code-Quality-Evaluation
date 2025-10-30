from typing import List

class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        """
        Finds the maximum absolute difference between adjacent elements in a circular array.

        Args:
            nums: A list of integers representing a circular array.

        Returns:
            The maximum absolute difference between adjacent elements.
        """
        n = len(nums)
        if n < 2:
            # Although constraints say n >= 2, handle for robustness
            return 0

        max_diff = 0

        # Iterate through adjacent pairs in the linear sense (i, i+1)
        for i in range(n - 1):
            diff = abs(nums[i+1] - nums[i])
            max_diff = max(max_diff, diff)

        # Check the wrap-around adjacent pair (last element, first element)
        circular_diff = abs(nums[0] - nums[n-1])
        max_diff = max(max_diff, circular_diff)

        return max_diff