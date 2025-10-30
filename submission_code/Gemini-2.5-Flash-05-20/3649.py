import math
import itertools
from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        
        # Initialize min_total_time to a very large value
        min_total_time = float('inf')
        
        # Generate all permutations of lock indices (0 to n-1)
        # Each permutation represents a possible order in which to break the locks
        # For n=8, there are 8! = 40,320 permutations, which is computationally feasible.
        for p in itertools.permutations(range(n)):
            current_time_for_permutation = 0
            current_X = 1  # Initial factor X for the first lock
            
            # Iterate through the locks in the current permutation order
            # 'i' represents the sequence number of the lock being broken (0th, 1st, 2nd, ...)
            # 'lock_index' is the original index of the lock in the 'strength' array
            for i in range(n):
                lock_index = p[i]
                s_val = strength[lock_index] # Energy required for this specific lock
                
                # Calculate minutes needed to break this lock.
                # The sword energy increases by 'current_X' per minute.
                # To reach 's_val' energy, it takes ceil(s_val / current_X) minutes.
                # Using integer division: ceil(A/B) can be calculated as (A + B - 1) // B for positive integers A, B.
                minutes_for_this_lock = (s_val + current_X - 1) // current_X
                
                # Add the minutes taken for this lock to the total time for this specific permutation
                current_time_for_permutation += minutes_for_this_lock
                
                # After breaking a lock, the sword energy implicitly resets to 0 (as we only accumulate time),
                # and the factor X increases by K for the *next* lock.
                current_X += K
            
            # Update the overall minimum time found across all permutations
            min_total_time = min(min_total_time, current_time_for_permutation)
            
        return min_total_time