import itertools
from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        min_time = float('inf')
        
        # Iterate through all possible permutations of the strength array
        for perm in itertools.permutations(strength):
            total = 0
            for i in range(n):
                x = 1 + K * i  # Current factor X after i increments of K
                s = perm[i]
                # Calculate the time needed to break the current lock
                t = (s + x - 1) // x  # Equivalent to ceiling division
                total += t
                # Early exit if current total exceeds the current minimum
                if total >= min_time:
                    break
            # Update the minimum time if current permutation's total is smaller
            if total < min_time:
                min_time = total
        
        return min_time