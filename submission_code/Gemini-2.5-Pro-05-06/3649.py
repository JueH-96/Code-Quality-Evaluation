import math
import itertools

class Solution:
  def findMinimumTime(self, strength: list[int], K: int) -> int:
    n = len(strength)
    
    # `indices` will store the original indices of the locks, e.g., [0, 1, ..., n-1].
    # We will generate permutations of these indices to decide the order of breaking locks.
    indices = list(range(n)) 

    min_total_time = float('inf')

    # Iterate through all possible orders (permutations) of breaking locks.
    # For n up to 8, n! is at most 8! = 40320.
    # Each permutation processing takes O(n) time. Total complexity O(n * n!).
    for p in itertools.permutations(indices):
        # `p` is a tuple representing an order of lock indices.
        # For example, if n=3, strength=[s0,s1,s2], and p=(1,0,2), it means:
        # 1st break lock with strength[1], then 2nd break lock with strength[0], 
        # finally 3rd break lock with strength[2].
        
        current_total_time_for_this_permutation = 0
        
        # The factor X by which sword energy increases depends on how many locks
        # have already been broken.
        # After 0 locks are broken (i.e., for the 1st lock to break, i=0), factor X = 1 + 0*K = 1.
        # After 1 lock is broken (i.e., for the 2nd lock to break, i=1), factor X = 1 + 1*K.
        # After `i` locks are broken (i.e., for the (i+1)-th lock to break), factor X = 1 + i*K.
        
        for i in range(n):
            # `i` denotes that `i` locks have already been broken in this permutation.
            # So, we are about to break the (i+1)-th lock in this sequence.
            
            factor_X = 1 + i * K
            
            # Get the original index (in the `strength` array) of the lock we are breaking at this step.
            # p[i] gives this original index.
            lock_original_idx = p[i]
            lock_s = strength[lock_original_idx]
            
            # Time to charge sword for this specific lock: ceil(lock_s / factor_X).
            # Sword energy increases by factor_X each minute. Energy needed is lock_s.
            # So, t * factor_X >= lock_s  => t >= lock_s / factor_X. Min integer t is ceil(lock_s / factor_X).
            
            # Using integer arithmetic for ceil(A/B): (A + B - 1) // B for positive integers A, B.
            # Constraints: strength[j] >= 1. K >= 1 implies factor_X starts at 1 and increases.
            # So lock_s and factor_X are positive integers.
            time_for_this_lock = (lock_s + factor_X - 1) // factor_X
            
            current_total_time_for_this_permutation += time_for_this_lock
        
        # After calculating total time for the current permutation,
        # update the overall minimum time found so far.
        if current_total_time_for_this_permutation < min_total_time:
            min_total_time = current_total_time_for_this_permutation
            
    # Constraints: n >= 1, so min_total_time will surely be updated from float('inf').
    # All time calculations involve integers, so the sum (total time) will be an integer.
    return int(min_total_time)