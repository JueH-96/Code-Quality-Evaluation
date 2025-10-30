import math
import itertools

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        min_time = float('inf')
        
        for perm in itertools.permutations(range(n)):
            current_time = 0
            X = 1  # Initial X
            for idx in perm:
                time = math.ceil(strength[idx] / X)
                current_time += time
                X += K
            min_time = min(min_time, current_time)
        
        return min_time