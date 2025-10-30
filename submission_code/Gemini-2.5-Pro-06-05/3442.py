from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        """
        Calculates the maximum total reward that can be collected.

        The problem is solved using dynamic programming with a bitmask. We determine the set of all
        achievable total rewards by iterating through the unique sorted reward values.

        Args:
            rewardValues: A list of integers representing the values of rewards.

        Returns:
            An integer denoting the maximum total reward.
        """
        
        # Step 1: Sort and get unique reward values.
        # A reward value `v` can be taken at most once. If we take it with current sum `x`,
        # the new sum is `x+v`. To take `v` again, we'd need `v > x+v`, which is impossible.
        # Sorting helps process rewards in a structured manner.
        rewards = sorted(list(set(rewardValues)))
        
        # Step 2: Use a bitmask to represent the set of reachable sums.
        # `reachable_sums` is an integer where the i-th bit is 1 if sum `i` is reachable.
        # Initially, only a sum of 0 is reachable (by taking no rewards).
        reachable_sums = 1
        
        # Step 3: Iterate through each unique reward and update the set of reachable sums.
        for r in rewards:
            # For the current reward `r`, we can add it to any previously achievable sum `x`
            # as long as the condition `r > x` is met.
            
            # To find all achievable sums `x < r`, we create a mask.
            # `(1 << r) - 1` creates a number with its first `r` bits set to 1.
            mask = (1 << r) - 1
            
            # Isolate the bits in `reachable_sums` for sums less than `r`.
            sums_less_than_r = reachable_sums & mask
            
            # For each sum `x` in `sums_less_than_r`, we can now form `x + r`.
            # This is equivalent to left-shifting the bitmask by `r`.
            newly_reachable_sums = sums_less_than_r << r
            
            # Add these new sums to our set of reachable sums.
            reachable_sums |= newly_reachable_sums
            
        # Step 4: The maximum reward is the position of the most significant bit.
        # `int.bit_length()` for a positive integer `n` gives `floor(log2(n)) + 1`.
        # If the highest bit is at position `p` (value 2^p), bit_length is `p+1`.
        # So, the maximum sum `p` is `reachable_sums.bit_length() - 1`.
        return reachable_sums.bit_length() - 1