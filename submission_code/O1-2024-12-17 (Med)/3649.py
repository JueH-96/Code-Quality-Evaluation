class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        from itertools import permutations
        import math
        
        n = len(strength)
        min_time = float('inf')
        
        # Try all permutations of the lock-breaking order
        for order in permutations(strength):
            total_minutes = 0
            
            # For the p-th lock in the order, factor = 1 + (p-1)*K
            for p, required_energy in enumerate(order, start=1):
                factor = 1 + (p - 1) * K
                # Time to break current lock = ceil(required_energy / factor)
                total_minutes += (required_energy + factor - 1) // factor
            
            min_time = min(min_time, total_minutes)
        
        return min_time