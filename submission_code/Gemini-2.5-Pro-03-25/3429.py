import math
from typing import List

class Solution:
    """
    Determines if an array is "special".
    An array is special if every pair of its adjacent elements contains two numbers with different parity.
    """
    def isArraySpecial(self, nums: List[int]) -> bool:
        """
        Checks if the given array `nums` is special.

        Args:
            nums: A list of integers.

        Returns:
            True if the array is special, False otherwise.
        """
        n = len(nums)
        
        # An array with 0 or 1 element has no adjacent pairs, 
        # so the condition is vacuously true.
        if n <= 1:
            return True
            
        # Iterate through all adjacent pairs
        # We only need to iterate up to the second-to-last element
        for i in range(n - 1):
            # Check the parity of the current element and the next element.
            # Parity is determined by the remainder when divided by 2.
            # If both remainders are the same (both 0 or both 1), 
            # they have the same parity.
            if nums[i] % 2 == nums[i+1] % 2:
                # Found a pair with the same parity, so the array is not special.
                return False
                
        # If the loop completes without returning False, it means all adjacent 
        # pairs have different parity.
        return True