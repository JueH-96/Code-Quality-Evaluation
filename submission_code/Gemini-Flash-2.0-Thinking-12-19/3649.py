import itertools

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        min_time = float('inf')
        indices = list(range(n))
        permutations = list(itertools.permutations(indices))

        for perm in permutations:
            current_time = 0
            current_energy = 0
            current_X = 1
            for lock_index in perm:
                required_strength = strength[lock_index]
                while current_energy < required_strength:
                    current_time += 1
                    current_energy += current_X
                current_energy = 0
                current_X += K
            min_time = min(min_time, current_time)
        return min_time