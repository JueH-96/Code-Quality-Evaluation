class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        import math
        from itertools import permutations
        
        n = len(strength)
        min_time = float('inf')
        
        # Try all possible permutations of breaking the locks
        for perm in permutations(range(n)):
            time = 0
            X = 1
            
            # Break locks in this order
            for i in perm:
                # Time needed to break this lock with current X
                time_needed = math.ceil(strength[i] / X)
                time += time_needed
                # After breaking, X increases by K
                X += K
            
            min_time = min(min_time, time)
        
        return min_time