from typing import List
from itertools import permutations

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        valid_numbers = set()
        # generate all 3-length permutations
        for p in permutations(digits, 3):
            # leading zero check: first digit should not be zero
            if p[0] == 0:
                continue
            # even number check: last digit should be even
            if p[-1] % 2 != 0:
                continue
            # convert the tuple into a number
            num = p[0] * 100 + p[1] * 10 + p[2]
            valid_numbers.add(num)
        return len(valid_numbers)