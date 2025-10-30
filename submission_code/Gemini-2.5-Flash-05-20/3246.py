from typing import List

class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        """
        Checks if it's possible to select two or more elements in the array
        such that the bitwise OR of the selected elements has at least one
        trailing zero in its binary representation.

        A number has at least one trailing zero in its binary representation
        if and only if it is an even number (its least significant bit is 0).

        The bitwise OR of a set of numbers will be even (have a trailing zero)
        if and only if all numbers in that set are themselves even. This is
        because the least significant bit of a bitwise OR is the OR of the
        least significant bits of the operands. To get a 0 (even) for the LSB
        of the result, all LSBs of the operands must be 0.

        Therefore, the problem simplifies to checking if there are at least
        two even numbers in the input array `nums`.

        Args:
            nums: A list of positive integers.

        Returns:
            True if it's possible to select two or more even numbers from `nums`,
            False otherwise.
        """
        even_count = 0
        for num in nums:
            if num % 2 == 0:  # Check if the number is even
                even_count += 1
            
            # If we have found two or more even numbers, we can satisfy the
            # condition by selecting those two (or more). Their bitwise OR
            # will be even. So, we can return True immediately.
            if even_count >= 2:
                return True
        
        # If the loop completes and even_count is less than 2, it means we
        # could not find two or more even numbers. In this case, any
        # selection of two or more elements will involve at least one odd
        # number, making their bitwise OR odd (no trailing zeros).
        return False