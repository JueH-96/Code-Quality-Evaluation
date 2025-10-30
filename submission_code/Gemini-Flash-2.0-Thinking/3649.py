from itertools import permutations
from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        min_time = float('inf')

        for perm in permutations(strength):
            current_time = 0
            current_factor = 1
            for strength_needed in perm:
                time_to_break = (strength_needed + current_factor - 1) // current_factor
                current_time += time_to_break
                current_factor += K
            min_time = min(min_time, current_time)

        return min_time