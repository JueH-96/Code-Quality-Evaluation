from typing import List
import itertools

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_val = 0
        # Try all 6 permutations of the three numbers
        for perm in itertools.permutations(nums):
            # Build the concatenated binary string
            bits = ''.join(bin(x)[2:] for x in perm)
            # Convert to integer and update maximum
            val = int(bits, 2)
            if val > max_val:
                max_val = val
        return max_val