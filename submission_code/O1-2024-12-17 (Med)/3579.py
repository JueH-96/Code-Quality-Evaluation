from typing import List
from itertools import permutations

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_val = 0
        
        # Generate all permutations of the 3 numbers
        for perm in permutations(nums, 3):
            # Convert each number to binary (without leading zeros) and concatenate
            binary_str = ''.join(bin(x)[2:] for x in perm)
            
            # Convert the concatenated binary string to an integer
            val = int(binary_str, 2)
            
            # Keep track of the maximum value
            max_val = max(max_val, val)
        
        return max_val