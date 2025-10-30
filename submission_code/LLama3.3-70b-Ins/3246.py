from typing import List
import itertools

class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # Generate all possible combinations of two or more elements
        for r in range(2, len(nums) + 1):
            for combination in itertools.combinations(nums, r):
                # Calculate the bitwise OR of the current combination
                bitwise_or = 0
                for num in combination:
                    bitwise_or |= num
                
                # Check if the bitwise OR has at least one trailing zero
                if bitwise_or % 2 == 0:
                    return True
        
        # If no combination has a bitwise OR with trailing zeros, return False
        return False