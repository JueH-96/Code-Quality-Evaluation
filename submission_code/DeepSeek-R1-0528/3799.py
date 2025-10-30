from itertools import permutations
from typing import List

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        s = set()
        for perm in permutations(digits, 3):
            if perm[0] == 0:
                continue
            if perm[2] % 2 == 0:
                s.add(perm)
        return len(s)