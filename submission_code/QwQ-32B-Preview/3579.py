from typing import List
import itertools

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_value = 0
        # Generate all permutations of the three numbers
        for perm in itertools.permutations(nums):
            # Concatenate binary representations
            binary_str = ''.join(bin(x)[2:] for x in perm)
            # Convert to integer
            value = int(binary_str, 2)
            # Update max_value if this value is larger
            if value > max_value:
                max_value = value
        return max_value