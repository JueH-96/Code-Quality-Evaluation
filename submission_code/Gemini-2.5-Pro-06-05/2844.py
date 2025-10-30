from typing import List

class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        """
        Calculates the sum of squares of special elements in a list.

        An element is special if its 1-based index divides the length of the list.
        
        Args:
            nums: A list of integers. The problem statement considers this 1-indexed.

        Returns:
            The sum of the squares of all special elements.
        """
        n = len(nums)
        
        # We iterate through the list with 1-based indexing using enumerate(nums, 1).
        # For each element, if its index 'i' divides 'n', we include its square
        # in the sum. A generator expression makes this very concise.
        return sum(val * val for i, val in enumerate(nums, 1) if n % i == 0)