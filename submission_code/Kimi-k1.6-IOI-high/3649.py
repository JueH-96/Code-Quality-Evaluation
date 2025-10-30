import itertools
from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        min_time = float('inf')
        for perm in itertools.permutations(strength):
            total = 0
            for i in range(n):
                x = 1 + i * K
                total += (perm[i] + x - 1) // x  # Ceiling division
            if total < min_time:
                min_time = total
        return min_time