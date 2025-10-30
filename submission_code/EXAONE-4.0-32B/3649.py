import itertools
from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        min_time = float('inf')
        for perm in itertools.permutations(range(n)):
            total_time = 0
            for j in range(n):
                factor = 1 + j * K
                s = strength[perm[j]]
                t = (s + factor - 1) // factor
                total_time += t
            if total_time < min_time:
                min_time = total_time
        return min_time