from typing import List

class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # For the bitwise OR to have a trailing zero, its least significant bit must be 0.
        # In binary, an even number always ends with 0. Since bitwise OR of numbers
        # is computed bit-by-bit (and OR of a set of numbers is 0 in a particular bit only if all
        # the numbers have a 0 in that bit), the final result is even if and only if every chosen number is even.
        # However, the problem requires selecting at least two elements.
        # Therefore, we simply check if there are at least two even numbers in the input.
        
        even_count = sum(1 for num in nums if num % 2 == 0)
        return even_count >= 2