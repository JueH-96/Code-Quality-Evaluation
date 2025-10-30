from typing import List
import itertools

class Solution:
    def maximumGood(self, nums: List[int]) -> int:
        # Generate all permutations of the input list
        perms = list(itertools.permutations(nums))
        
        max_num = 0
        
        # Iterate over each permutation
        for perm in perms:
            # Convert each number to binary, remove the '0b' prefix, and concatenate them
            binary_str = ''.join(bin(num)[2:] for num in perm)
            
            # Convert the binary string back to an integer
            num = int(binary_str, 2)
            
            # Update the maximum number if necessary
            max_num = max(max_num, num)
        
        return max_num