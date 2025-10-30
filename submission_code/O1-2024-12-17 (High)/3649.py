class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        from itertools import permutations
        
        n = len(strength)
        min_time = float('inf')
        
        # Try all possible orders of breaking the locks
        for order in permutations(strength):
            time_needed = 0
            X = 1   # Initial factor
            
            for s in order:
                # Number of minutes needed to get energy >= s
                wait = (s + X - 1) // X
                time_needed += wait
                # After breaking a lock, sword energy resets to 0, factor increments by K
                X += K
            
            min_time = min(min_time, time_needed)
        
        return min_time