from typing import List
from itertools import permutations

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_number = 0
        
        for perm in permutations(nums):
            # Concatenate the binary representations of the numbers in the current permutation
            binary_string = ''.join(bin(num)[2:] for num in perm)
            # Convert the binary string to an integer
            number = int(binary_string, 2)
            # Update the maximum number found
            max_number = max(max_number, number)
        
        return max_number