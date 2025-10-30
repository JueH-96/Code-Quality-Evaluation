from typing import List

class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # A bitwise OR will have its least significant bit (LSB) = 0
        # (i.e., be even) if and only if all selected numbers have LSB = 0 (are even).
        # Since we need at least two elements, we just check if there are
        # at least two even numbers in the array.
        evens = sum(1 for x in nums if x % 2 == 0)
        return evens >= 2