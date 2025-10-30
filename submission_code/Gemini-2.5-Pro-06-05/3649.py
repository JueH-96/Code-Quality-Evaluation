from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        
        # dp[mask] will store the minimum time to break the locks represented by the bitmask 'mask'.
        # A mask is an integer where the i-th bit is 1 if lock i has been broken.
        # This problem is equivalent to finding an optimal permutation of locks to break, which
        # can be solved with dynamic programming on subsets (bitmask DP).
        dp = [float('inf')] * (1 << n)
        
        # Base case: 0 time to break 0 locks.
        dp[0] = 0
        
        # Iterate through all possible subsets of locks (represented by masks).
        # The order of masks from 1 upwards ensures that when we calculate dp[mask],
        # dp[prev_mask] (where prev_mask is a submask of mask) is already computed.
        for mask in range(1, 1 << n):
            # The number of locks broken in the current subset determines the factor X.
            # This can be found by counting the set bits in the mask (population count).
            num_broken = bin(mask).count('1')
            
            # For the k-th lock broken, the factor is 1 + (k-1)*K.
            factor_X = 1 + (num_broken - 1) * K
            
            # To compute dp[mask], we consider which lock was the *last* one to be broken
            # to form this subset. Iterate through each lock to see if it's part of the current subset.
            for j in range(n):
                # Check if the j-th lock is in the current subset (mask).
                if (mask >> j) & 1:
                    # If lock 'j' is in the set, it could have been the last one broken.
                    # The state before breaking lock 'j' would be 'prev_mask'.
                    prev_mask = mask ^ (1 << j)
                    
                    # Calculate the time required to break lock 'j' with the current factor.
                    # This is ceil(strength[j] / factor_X).
                    # Using integer arithmetic: (numerator + denominator - 1) // denominator.
                    time_for_this_lock = (strength[j] + factor_X - 1) // factor_X
                    
                    # The total time for this sequence of breaks is the time for the previous
                    # subset plus the time for this last lock.
                    current_total_time = dp[prev_mask] + time_for_this_lock
                    
                    # We want the minimum time to reach the state 'mask', so we take the minimum
                    # over all possible choices for the last lock.
                    dp[mask] = min(dp[mask], current_total_time)
                    
        # The final answer is the minimum time to break all locks (mask with all bits set).
        return dp[(1 << n) - 1]