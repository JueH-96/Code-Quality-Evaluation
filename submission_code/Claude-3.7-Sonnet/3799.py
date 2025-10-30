from itertools import permutations
from typing import List

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        # Create a set to store distinct valid numbers
        valid_numbers = set()
        
        # Generate all permutations of 3 digits
        for perm in permutations(digits, 3):
            # Check if it's a valid 3-digit even number
            if perm[0] != 0 and perm[2] % 2 == 0:
                num = perm[0] * 100 + perm[1] * 10 + perm[2]
                valid_numbers.add(num)
        
        return len(valid_numbers)