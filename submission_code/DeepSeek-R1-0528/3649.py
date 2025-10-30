from itertools import permutations

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        min_time = float('inf')
        
        for perm in permutations(strength):
            total_time = 0
            for j, s in enumerate(perm):
                X = 1 + j * K
                total_time += (s + X - 1) // X
            if total_time < min_time:
                min_time = total_time
        
        return min_time