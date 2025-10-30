from typing import List

class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        """
        Finds the maximum absolute difference between adjacent elements in a circular array.

        Args:
            nums: A list of integers representing the circular array.

        Returns:
            The maximum absolute difference between adjacent elements.
        """
        n = len(nums)
        if n < 2:
            # Constraint says 2 <= nums.length, but good practice to handle edge cases
            return 0 # Or raise an error depending on desired behavior for invalid input

        # Initialize max_diff with the difference between the first two elements
        # We will iterate through all adjacent pairs and update this value.
        max_diff = 0 # Initialize with 0 since absolute differences are non-negative

        # Iterate through standard adjacent pairs: (nums[i], nums[i+1])
        for i in range(n - 1):
            diff = abs(nums[i] - nums[i+1])
            max_diff = max(max_diff, diff)

        # Consider the circular adjacent pair: (nums[n-1], nums[0])
        circular_diff = abs(nums[n-1] - nums[0])
        max_diff = max(max_diff, circular_diff)

        return max_diff