from typing import List
from itertools import permutations

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        # Set to store unique three-digit even numbers
        unique_numbers = set()
        
        # Generate all permutations of length 3
        for perm in permutations(digits, 3):
            # Form the number from the permutation
            number = perm[0] * 100 + perm[1] * 10 + perm[2]
            
            # Check if it's a valid three-digit even number
            if perm[0] != 0 and perm[2] % 2 == 0:
                unique_numbers.add(number)
        
        # Return the count of unique numbers
        return len(unique_numbers)