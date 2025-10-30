from itertools import permutations

class Solution:
    def findMinimumTime(self, strength, K):
        min_time = float('inf')
        for perm in permutations(strength):
            x = 1
            total = 0
            for s in perm:
                t = (s + x - 1) // x
                total += t
                x += K
            if total < min_time:
                min_time = total
        return min_time