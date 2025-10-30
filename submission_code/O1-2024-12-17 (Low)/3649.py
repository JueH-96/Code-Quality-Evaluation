class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        import math
        from itertools import permutations
        
        n = len(strength)
        min_time = float('inf')
        
        # Try all permutations of the locks
        for perm in permutations(strength):
            current_time = 0
            X = 1  # initial factor
            # Simulate the process for this permutation
            for required_energy in perm:
                # time needed to reach required_energy with factor X
                needed = math.ceil(required_energy / X)
                current_time += needed
                # after breaking the lock
                X += K  # increase factor by K
                
            min_time = min(min_time, current_time)
        
        return min_time