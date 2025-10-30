class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        from itertools import permutations
        import math
        
        n = len(strength)
        min_time = float('inf')
        
        # Try all possible orders of breaking locks
        for perm in permutations(range(n)):
            total_time = 0
            current_X = 1
            
            for lock_idx in perm:
                # Time needed to break this lock = ceil(strength[lock_idx] / current_X)
                time_needed = math.ceil(strength[lock_idx] / current_X)
                total_time += time_needed
                
                # After breaking the lock, X increases by K
                current_X += K
            
            min_time = min(min_time, total_time)
        
        return min_time