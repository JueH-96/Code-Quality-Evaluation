import itertools

class Solution:
    def findMinimumTime(self, strength: list, K: int) -> int:
        n = len(strength)
        min_time = float('inf')
        for perm in itertools.permutations(range(n)):
            current_X = 1
            total = 0
            for i in perm:
                s = strength[i]
                t = (s + current_X - 1) // current_X  # Equivalent to ceil(s / current_X)
                total += t
                current_X += K
            if total < min_time:
                min_time = total
        return min_time