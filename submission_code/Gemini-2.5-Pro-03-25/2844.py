import math
from typing import List

class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        """
        Calculates the sum of the squares of all special elements in a 1-indexed array.
        An element nums[i] is special if its 1-based index i divides the length of the array n.

        Args:
            nums: A 1-indexed (conceptually) list of integers. In Python, it's a 0-indexed list.

        Returns:
            The sum of the squares of all special elements.
        """
        
        n = len(nums)
        total_sum_of_squares = 0
        
        # Iterate through the indices of the array. Since the problem defines 
        # indices as 1-based, we iterate from 1 to n (inclusive).
        for i in range(1, n + 1):
            # Check if the 1-based index 'i' divides the length 'n'
            if n % i == 0:
                # If 'i' divides 'n', then the element at this 1-based index is special.
                # Remember that Python lists are 0-indexed, so the element corresponding 
                # to the 1-based index 'i' is located at index 'i-1' in the list 'nums'.
                element = nums[i-1]
                
                # Calculate the square of the special element
                square = element * element
                
                # Add the square to the total sum
                total_sum_of_squares += square
                
        return total_sum_of_squares