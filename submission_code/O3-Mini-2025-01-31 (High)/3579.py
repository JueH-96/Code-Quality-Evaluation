from itertools import permutations
from typing import List

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        best = 0
        # Try all permutations of the 3 numbers.
        for perm in permutations(nums):
            # Create the binary string by concatenating each number's binary representation (without the "0b" prefix).
            binary_str = "".join(format(num, 'b') for num in perm)
            # Convert the concatenated binary string to an integer.
            value = int(binary_str, 2)
            best = max(best, value)
        return best