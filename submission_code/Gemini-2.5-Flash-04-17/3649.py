from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        # dp[mask] stores the minimum time required to break the subset of locks represented by the set bits in 'mask'.
        # mask is an integer from 0 to 2^n - 1. The i-th bit corresponds to the i-th lock (0-indexed).
        
        dp_size = 1 << n 
        
        # Initialize dp table: infinite time for all states except the initial state (no locks broken).
        # Using a large integer like float('inf') is suitable for finding the minimum.
        dp = [float('inf')] * dp_size
        dp[0] = 0 # Time to break 0 locks is 0.
        
        # Iterate through all possible states (masks) starting from the empty set (mask=0).
        # By iterating mask from 0 to 2^n - 1, we process states with fewer broken locks
        # before states with more broken locks. This ensures that when we consider
        # transitioning from 'mask' to 'next_mask = mask | (1 << j)', the value dp[mask]
        # has already been finalized with the minimum possible time to reach that state.
        for mask in range(dp_size):
            # If the current state is unreachable (has infinite time), we cannot transition from it.
            if dp[mask] == float('inf'):
                continue
            
            # Count the number of locks already broken in this state (number of set bits in mask).
            # This count determines the factor X for the next lock to be broken.
            # The number of set bits in the mask equals the number of broken locks.
            num_broken = bin(mask).count('1')
            
            # The factor X for the (num_broken + 1)-th lock to be broken is 1 + num_broken * K.
            current_factor_X = 1 + num_broken * K
            
            # Consider breaking each lock that is not yet broken in the current state (mask).
            for j in range(n):
                # Check if the j-th lock is not broken (j-th bit is 0 in mask).
                if not (mask >> j) & 1:
                    # Calculate the time required to break lock j, given the current factor X.
                    required_strength = strength[j]
                    
                    # Time needed = ceil(required_strength / current_factor_X).
                    # Using integer division for ceiling: (a + b - 1) // b for positive a, b.
                    # required_strength is >= 1 and current_factor_X = 1 + num_broken * K >= 1,
                    # so this formula is valid.
                    time_needed = (required_strength + current_factor_X - 1) // current_factor_X
                    
                    # The state after breaking lock j is mask with the j-th bit set.
                    next_mask = mask | (1 << j)
                    
                    # Update the minimum time to reach the next state (next_mask).
                    # The time to reach next_mask by breaking lock j is the time to reach mask
                    # plus the time needed to break lock j.
                    dp[next_mask] = min(dp[next_mask], dp[mask] + time_needed)
        
        # The final state where all locks are broken is represented by the mask with all bits set (2^n - 1).
        # dp[2^n - 1] stores the minimum time required to reach this state.
        return dp[dp_size - 1]