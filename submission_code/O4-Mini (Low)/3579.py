from typing import List
import itertools

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_val = 0
        for perm in itertools.permutations(nums):
            # Concatenate binary representations without the '0b' prefix
            bin_str = ''.join(bin(x)[2:] for x in perm)
            # Convert back to integer
            value = int(bin_str, 2)
            if value > max_val:
                max_val = value
        return max_val