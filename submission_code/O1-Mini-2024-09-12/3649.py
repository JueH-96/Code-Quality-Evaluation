from typing import List
from itertools import permutations
import math

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        min_time = float('inf')
        
        # Generate all possible permutations of the locks
        for order in permutations(strength):
            total_time = 0
            current_X = 1  # Initial factor
            
            for s in order:
                # Calculate the time needed to break the current lock
                time_needed = math.ceil(s / current_X)
                total_time += time_needed
                
                # Update the factor X after breaking the lock
                current_X += K
            
            # Update the minimum time if current total_time is lower
            if total_time < min_time:
                min_time = total_time
        
        return min_time