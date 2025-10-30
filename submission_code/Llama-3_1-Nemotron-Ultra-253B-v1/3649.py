from itertools import permutations
from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        min_time = float('inf')
        for perm in permutations(range(n)):
            current_x = 1
            total = 0
            for idx in perm:
                s = strength[idx]
                t = (s + current_x - 1) // current_x  # Equivalent to ceil(s / current_x)
                total += t
                current_x += K
            if total < min_time:
                min_time = total
        return min_time