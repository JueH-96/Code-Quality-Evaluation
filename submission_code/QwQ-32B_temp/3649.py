import itertools
from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        min_time = float('inf')
        for perm in itertools.permutations(strength):
            current_time = 0
            x = 1  # initial factor
            for s in perm:
                # Calculate the required time for current lock
                t = (s + x - 1) // x  # equivalent to ceil(s / x)
                current_time += t
                x += K  # update the factor after breaking the lock
            if current_time < min_time:
                min_time = current_time
        return min_time