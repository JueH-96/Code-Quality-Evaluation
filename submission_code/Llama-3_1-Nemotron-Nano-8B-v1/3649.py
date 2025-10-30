import itertools
from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        min_time = float('inf')
        n = len(strength)
        for perm in itertools.permutations(strength):
            total = 0
            for i, s in enumerate(perm):
                x = 1 + i * K
                t = (s + x - 1) // x  # Equivalent to ceil(s / x)
                total += t
            if total < min_time:
                min_time = total
        return min_time