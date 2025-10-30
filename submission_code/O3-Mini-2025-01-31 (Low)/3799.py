from typing import List
import itertools

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        results = set()
        # generate all 3-digit arrangements (permutations) using indices
        for perm in itertools.permutations(digits, 3):
            # ensure no leading zero
            if perm[0] == 0:
                continue
            # check if last digit is even
            if perm[2] % 2 != 0:
                continue
            # convert tuple to integer number
            number = perm[0] * 100 + perm[1] * 10 + perm[2]
            results.add(number)
        return len(results)