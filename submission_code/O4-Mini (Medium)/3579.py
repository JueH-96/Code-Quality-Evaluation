from typing import List
import itertools

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_val = 0
        # Try all permutations of the three numbers
        for perm in itertools.permutations(nums):
            # Concatenate their binary representations (without the "0b" prefix)
            bin_str = ''.join(bin(x)[2:] for x in perm)
            # Convert back to integer
            val = int(bin_str, 2)
            max_val = max(max_val, val)
        return max_val