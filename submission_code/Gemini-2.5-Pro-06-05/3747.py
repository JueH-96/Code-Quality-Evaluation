from typing import List

class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        """
        Finds the maximum absolute difference between adjacent elements in a circular array.

        Args:
            nums: A list of integers representing a circular array. The length is at least 2.

        Returns:
            The maximum absolute difference between any two adjacent elements.
        """
        
        n = len(nums)
        
        # The constraints (2 <= nums.length) guarantee at least two elements.
        # Initialize the maximum difference with the difference between the last and first elements,
        # since the array is circular.
        max_diff = abs(nums[-1] - nums[0])
        
        # Iterate through the standard adjacent pairs (i.e., nums[0] & nums[1], nums[1] & nums[2], etc.).
        for i in range(n - 1):
            # Calculate the absolute difference for the current pair.
            current_diff = abs(nums[i] - nums[i+1])
            
            # Update the maximum difference if the current pair's difference is larger.
            max_diff = max(max_diff, current_diff)
            
        return max_diff