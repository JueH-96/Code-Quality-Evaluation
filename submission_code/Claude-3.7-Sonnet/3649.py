class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        min_time = float('inf')
        
        def dfs(mask, time, energy, X):
            nonlocal min_time
            if mask == 0:  # All locks broken
                min_time = min(min_time, time)
                return
            
            if time >= min_time:  # Pruning
                return
            
            for i in range(n):
                if (mask >> i) & 1:  # If lock i is not broken
                    if energy < strength[i]:
                        # Calculate additional time needed to break this lock
                        additional_time = (strength[i] - energy + X - 1) // X  # Ceiling division
                        new_time = time + additional_time
                        new_energy = energy + additional_time * X
                    else:
                        # Already have enough energy to break this lock
                        new_time = time
                        new_energy = energy
                    
                    new_mask = mask & ~(1 << i)  # Remove lock i from the mask
                    dfs(new_mask, new_time, 0, X + K)  # Energy resets to 0 after breaking a lock
        
        dfs((1 << n) - 1, 0, 0, 1)  # Start with all locks unbroken
        return min_time