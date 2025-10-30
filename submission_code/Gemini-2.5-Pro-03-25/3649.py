import itertools
from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        """
        Calculates the minimum time required to break all locks in the dungeon
        by trying all possible permutations of lock breaking order.

        The time required depends on the order in which locks are broken because
        the factor 'X' (which determines how fast the sword energy increases)
        increases after each lock is broken.

        Args:
            strength: A list of integers where strength[i] is the energy needed 
                      to break the i-th lock.
            K: The value by which the sword's energy factor X increases after 
               breaking a lock.

        Returns:
            The minimum time in minutes required to break all n locks. Returns an integer.
        """
        
        # Determine the number of locks
        n = len(strength)
        
        # Initialize the minimum total time found so far to positive infinity.
        # This ensures that the first calculated total time will become the minimum.
        min_total_time = float('inf')

        # Create a list of lock indices [0, 1, ..., n-1]
        # These indices will be permuted to represent different orders of breaking locks.
        indices = list(range(n))
        
        # Generate all possible permutations of the lock indices.
        # Each permutation represents a unique sequence in which the locks can be broken.
        # For n locks, there are n! permutations. Given n <= 8, n! is at most 8! = 40320,
        # which is computationally feasible.
        all_permutations = itertools.permutations(indices)

        # Iterate through each possible order (permutation) of breaking locks
        for p in all_permutations:
            # For each permutation, simulate the process and calculate the total time required.
            
            # Initialize the total time for the current permutation sequence
            current_total_time = 0
            # Initialize the sword energy factor X. It starts at 1.
            current_X = 1 
            
            # Simulate breaking locks one by one according to the order in the current permutation 'p'
            for lock_idx in p:
                # Get the required energy (strength) for the current lock
                target_strength = strength[lock_idx]
                
                # Calculate the time needed to charge the sword enough to break this lock.
                # The sword energy starts at 0 after breaking the previous lock (or at time 0).
                # Every minute, the energy increases by the current factor `current_X`.
                # We need the minimum integer time 't' such that `t * current_X >= target_strength`.
                # This time `t` is equivalent to `ceil(target_strength / current_X)`.
                
                # We use the integer division formula for ceiling: (numerator + denominator - 1) // denominator
                # This avoids floating-point arithmetic and potential precision issues.
                # Note: Constraints ensure target_strength >= 1 and K >= 1. 
                # Since X starts at 1 and K >= 1, X will always be positive (>= 1).
                # Thus, we don't need to worry about division by zero or non-positive X.
                time_for_lock = (target_strength + current_X - 1) // current_X
                
                # Add the time taken for this lock to the total time for this permutation sequence
                current_total_time += time_for_lock
                
                # After successfully breaking the lock, the factor X increases by K.
                # This increased factor will be used for charging the sword for the next lock.
                current_X += K
                
            # After simulating the entire sequence for the current permutation,
            # compare its total time with the minimum time found so far and update if necessary.
            min_total_time = min(min_total_time, current_total_time)
            
        # After checking all permutations, min_total_time will hold the minimum possible time.
        # Return this minimum time.
        return min_total_time