from typing import List

class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # A bitwise OR has at least one trailing zero
        # if and only if its least significant bit is 0.
        # The LSB of the OR is the OR of all LSBs,
        # so it is 0 exactly when all selected numbers are even.
        # Hence we just need at least two even numbers.
        even_count = sum(1 for x in nums if x % 2 == 0)
        return even_count >= 2