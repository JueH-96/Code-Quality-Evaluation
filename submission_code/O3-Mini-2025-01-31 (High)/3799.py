from typing import List
from itertools import permutations

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        distinct_numbers = set()
        # Generate all 3-length permutations
        for perm in permutations(digits, 3):
            # Skip numbers with a leading zero
            if perm[0] == 0:
                continue
            # Last digit must be even
            if perm[2] % 2 != 0:
                continue
            # Form the number from the permutation tuple
            number = perm[0]*100 + perm[1]*10 + perm[2]
            distinct_numbers.add(number)
        return len(distinct_numbers)