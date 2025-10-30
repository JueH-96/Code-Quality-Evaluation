from itertools import permutations
from typing import List

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        distinct_numbers = set()
        
        # Generate all permutations of length 3
        for perm in permutations(digits, 3):
            # Check if the number is even and does not have leading zero
            if perm[2] % 2 == 0 and perm[0] != 0:
                distinct_numbers.add(perm)
        
        return len(distinct_numbers)