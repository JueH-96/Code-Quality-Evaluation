import math
import itertools

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        indices = list(range(n))
        min_time = float('inf')
        permutations = list(itertools.permutations(indices))
        for perm_indices in permutations:
            current_time = 0
            current_factor_x = 1
            for index in perm_indices:
                lock_strength = strength[index]
                time_to_break = (lock_strength + current_factor_x - 1) // current_factor_x
                current_time += time_to_break
                current_factor_x += K
            min_time = min(min_time, current_time)
        return min_time