from typing import List
import itertools

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        valid = set()
        for perm in itertools.permutations(digits, 3):
            if perm[0] == 0:
                continue
            if perm[2] % 2 != 0:
                continue
            number = perm[0] * 100 + perm[1] * 10 + perm[2]
            valid.add(number)
        return len(valid)