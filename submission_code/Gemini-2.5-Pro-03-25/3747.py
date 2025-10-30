import math
from typing import List

class Solution:
    """
    Finds the maximum absolute difference between adjacent elements in a circular array.
    """
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        """
        Calculates the maximum absolute difference between adjacent elements in a circular array.

        Args:
            nums: A list of integers representing the circular array.

        Returns:
            The maximum absolute difference found between any adjacent pair, 
            considering the first and last elements as adjacent.
        """
        n = len(nums)
        
        # Constraints state n >= 2, so we don't need to handle n < 2 explicitly,
        # but it's good practice. If n < 2, there are no adjacent pairs.
        if n < 2:
            return 0 
            
        max_diff = 0
        
        # Iterate through the array from the first element up to the second-to-last
        for i in range(n - 1):
            # Calculate the absolute difference between the current element and the next
            current_diff = abs(nums[i] - nums[i+1])
            # Update max_diff if the current difference is larger
            if current_diff > max_diff:
                max_diff = current_diff
                
        # Handle the circularity: calculate the difference between the last and first elements
        circular_diff = abs(nums[n-1] - nums[0])
        # Update max_diff if the circular difference is larger
        if circular_diff > max_diff:
            max_diff = circular_diff
            
        return max_diff

    # Alternative implementation using modulo operator for cleaner loop
    def maxAdjacentDistance_alternative(self, nums: List[int]) -> int:
        """
        Alternative implementation using modulo operator for cleaner loop.
        """
        n = len(nums)
        if n < 2:
            return 0
            
        max_diff = 0
        
        # Iterate through all elements
        for i in range(n):
            # Get the current element
            current_element = nums[i]
            # Get the next element index using modulo for wrap-around
            next_element_index = (i + 1) % n
            # Get the next element
            next_element = nums[next_element_index]
            
            # Calculate the absolute difference
            current_diff = abs(current_element - next_element)
            
            # Update the maximum difference found so far
            max_diff = max(max_diff, current_diff)
            
        return max_diff