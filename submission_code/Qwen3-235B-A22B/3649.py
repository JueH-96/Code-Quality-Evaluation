from typing import List
import itertools

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        min_time = float('inf')
        n = len(strength)
        # Generate all permutations of the strength list
        for perm in itertools.permutations(strength):
            total = 0
            x = 1  # initial X factor
            for s in perm:
                # Calculate the time needed to break this lock
                t = (s + x - 1) // x  # equivalent to ceil(s / x)
                total += t
                x += K  # increase X by K after breaking the lock
            if total < min_time:
                min_time = total
        return min_time