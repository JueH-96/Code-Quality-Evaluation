from typing import List

class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        """
        Checks if an array is special. An array is special if every pair of
        its adjacent elements contains two numbers with different parity.

        Args:
            nums: A list of integers.

        Returns:
            True if the array is special, False otherwise.
        """
        # An array with 0 or 1 elements is considered special as there are no adjacent pairs
        # to violate the condition. The loop below handles the single-element case correctly
        # because range(len(nums) - 1) will be range(0) for len=1, which is an empty range.
        # For len > 1, we check adjacent pairs.
        for i in range(len(nums) - 1):
            # Check if the parity of the current element is the same as the next element.
            # Parity is determined by the remainder when divided by 2.
            # If nums[i] % 2 == nums[i+1] % 2, they have the same parity (both even or both odd).
            if nums[i] % 2 == nums[i+1] % 2:
                # If we find any adjacent pair with the same parity, the array is not special.
                return False

        # If the loop completes without finding any adjacent elements with the same parity,
        # it means all adjacent pairs have different parities (or the array has only one element),
        # so the array is special.
        return True