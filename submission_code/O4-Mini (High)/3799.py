from typing import List
import itertools

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        valid_numbers = set()
        # Generate all 3-length permutations of the digits
        for a, b, c in itertools.permutations(digits, 3):
            # No leading zero and must be even
            if a != 0 and c % 2 == 0:
                num = 100 * a + 10 * b + c
                valid_numbers.add(num)
        return len(valid_numbers)