from itertools import permutations
from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        min_time = float('inf')
        for perm in permutations(strength):
            total = 0
            x = 1
            for s in perm:
                t = (s + x - 1) // x
                total += t
                x += K
            if total < min_time:
                min_time = total
        return min_time