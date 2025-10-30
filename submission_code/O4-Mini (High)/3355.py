from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        # Convert each level to a score: +1 if clearable, -1 if impossible
        total = sum(1 if v == 1 else -1 for v in possible)
        prefix = 0
        n = len(possible)
        
        # Alice must play at least 1 and at most n-1 levels
        # We look for the smallest k in [1, n-1] such that
        # Alice's score prefix > Bob's score (total - prefix)
        # i.e. 2 * prefix > total
        for i in range(n - 1):
            prefix += 1 if possible[i] == 1 else -1
            if 2 * prefix > total:
                return i + 1
        
        return -1