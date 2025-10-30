import itertools

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        min_time = float('inf')
        n = len(strength)
        for perm in itertools.permutations(strength):
            current_X = 1
            time = 0
            for s in perm:
                # Calculate the minimum time needed to reach strength s with current_X
                t = (s + current_X - 1) // current_X  # Equivalent to ceil(s / current_X)
                time += t
                current_X += K
            if time < min_time:
                min_time = time
        return min_time