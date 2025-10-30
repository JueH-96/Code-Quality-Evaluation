import collections
import bisect
from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        
        # Step 1: Count frequencies of each power value.
        counts = collections.Counter(power)

        # Step 2: Get the unique power values and sort them.
        unique_powers = sorted(counts.keys())
        
        n = len(unique_powers)
        if n == 0:
            return 0
        
        # dp[i] will store the maximum damage considering powers up to unique_powers[i].
        dp = [0] * n

        # Base case: For the first unique power, we must take it to get any damage.
        dp[0] = unique_powers[0] * counts[unique_powers[0]]

        # Fill the DP table by iterating through the unique powers.
        for i in range(1, n):
            current_power = unique_powers[i]
            
            # --- Option 1: Cast spells with current_power ---
            damage_from_current = current_power * counts[current_power]
            
            # Find the max damage from a non-conflicting previous state.
            # A previous power `p_prev` is non-conflicting if `p_prev < current_power - 2`.
            target = current_power - 2
            
            # Use binary search to find the index of the last power < target.
            # `bisect_left` finds the insertion point for `target`. All elements to the
            # left of this point are strictly less than `target`.
            # `hi=i` limits the search to `unique_powers[0...i-1]`.
            k = bisect.bisect_left(unique_powers, target, hi=i)
            
            prev_damage = 0
            if k > 0:
                # The index of the last valid previous state is k-1.
                # `dp[k-1]` stores the max damage up to that point.
                prev_damage = dp[k - 1]
            
            damage_if_take = damage_from_current + prev_damage
            
            # --- Option 2: Skip spells with current_power ---
            # The max damage is the same as the max from the previous state.
            damage_if_skip = dp[i-1]
            
            # The max damage at this step is the maximum of the two options.
            dp[i] = max(damage_if_take, damage_if_skip)

        # The final answer is the maximum damage considering all unique powers.
        return dp[-1]