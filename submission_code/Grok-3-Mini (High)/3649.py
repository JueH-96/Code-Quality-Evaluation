import itertools
from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        min_time = float('inf')
        for perm in itertools.permutations(strength):
            current_time = 0
            X = 1  # initial factor for the first lock
            for s in perm:
                # Calculate time to break the lock using ceiling division
                t = (s + X - 1) // X
                current_time += t
                # Increase X by K after breaking the lock
                X += K
            # Update minimum time
            if current_time < min_time:
                min_time = current_time
        return min_time