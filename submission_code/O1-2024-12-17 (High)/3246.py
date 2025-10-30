from typing import List

class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # We need at least two even numbers because the bitwise OR
        # of any subset that includes an odd number will be odd (last bit = 1).
        # Therefore, check if there are at least two even numbers in the list.
        even_count = sum(1 for x in nums if x % 2 == 0)
        return even_count >= 2