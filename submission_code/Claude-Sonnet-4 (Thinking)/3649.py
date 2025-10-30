class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        from itertools import permutations
        
        n = len(strength)
        min_time = float('inf')
        
        # Try all permutations
        for perm in permutations(range(n)):
            time = 0
            energy = 0
            X = 1
            
            for i in perm:
                # Calculate time needed to reach strength[i] energy
                if energy < strength[i]:
                    time_needed = (strength[i] - energy + X - 1) // X
                    time += time_needed
                    energy += time_needed * X
                
                # Break the lock
                energy = 0
                X += K
            
            min_time = min(min_time, time)
        
        return min_time