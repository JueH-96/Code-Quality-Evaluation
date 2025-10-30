from itertools import permutations

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        min_time = float('inf')
        
        # Generate all possible orders to break the locks
        for order in permutations(range(n)):
            current_time = 0
            energy = 0
            X = 1
            for i in order:
                # Calculate the time needed to reach the required energy
                # The energy increases by X each minute
                # We need to find the smallest t such that energy + X * t >= strength[i]
                # Since energy resets to 0 after breaking a lock, we can consider the energy as 0
                # So, we need X * t >= strength[i]
                # t >= ceil(strength[i] / X)
                t = (strength[i] + X - 1) // X
                current_time += t
                energy = 0
                X += K
            if current_time < min_time:
                min_time = current_time
        
        return min_time