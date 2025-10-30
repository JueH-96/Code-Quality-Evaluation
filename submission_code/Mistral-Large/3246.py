from typing import List

class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # To have trailing zeros in the binary representation, the bitwise OR of selected elements must be even.
        # This means at least one of the selected numbers must be even.
        # We need to check if there are at least two even numbers in the array.

        # Count the number of even numbers in the array
        even_count = sum(1 for num in nums if num % 2 == 0)

        # If there are at least two even numbers, we can select them to get a bitwise OR with trailing zeros
        if even_count >= 2:
            return true

        return false