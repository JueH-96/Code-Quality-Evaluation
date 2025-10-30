import collections
import bisect
from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        # Step 1: Count frequencies of each damage value.
        # Example: power = [1,1,3,4] -> counts = {1: 2, 3: 1, 4: 1}
        counts = collections.Counter(power)
        
        # Step 2: Calculate the total damage for each unique damage value.
        # This simplifies lookups later by pre-calculating value * count.
        # Example: total_damages_map = {1: 2, 3: 3, 4: 4}
        total_damages_map = {damage: damage * count for damage, count in counts.items()}
        
        # Step 3: Get unique damage values and sort them.
        # This order is crucial for our dynamic programming approach.
        # Example: sorted_unique_damages = [1, 3, 4]
        sorted_unique_damages = sorted(total_damages_map.keys())
        
        n = len(sorted_unique_damages)
        
        # If there are no spells, total damage is 0.
        if n == 0:
            return 0
            
        # Step 4: Initialize DP array.
        # dp[i] will store the maximum total damage considering spells
        # with damage values up to sorted_unique_damages[i].
        dp = [0] * n
        
        # Base case: For the first unique damage value (at index 0).
        # The maximum damage is simply its total value, as there are no previous spells to conflict with.
        dp[0] = total_damages_map[sorted_unique_damages[0]]
        
        # Step 5: Iterate through the sorted unique damage values to fill the DP array.
        for i in range(1, n):
            current_damage_val = sorted_unique_damages[i]
            current_total_damage = total_damages_map[current_damage_val]
            
            # Option 1: Do not cast any spell with current_damage_val.
            # The maximum damage is simply the max damage obtained considering values up to the previous index.
            take_none_current = dp[i-1]
            
            # Option 2: Cast all spells with current_damage_val.
            # If we cast spells with current_damage_val, we cannot cast any spells with
            # damage values: current_damage_val - 2, current_damage_val - 1,
            # current_damage_val + 1, or current_damage_val + 2.
            # For previous values, this means we must exclude any spells with
            # damage `current_damage_val - 1` or `current_damage_val - 2`.
            # So, we need to add the maximum damage from values that are strictly less than (current_damage_val - 2).
            
            # The threshold for previous compatible spells is (current_damage_val - 2).
            # We are looking for values D_prev such that D_prev < current_damage_val - 2.
            target_for_prev = current_damage_val - 2
            
            # Use bisect_left to find the insertion point for `target_for_prev`.
            # `prev_compatible_idx` will be the index of the first element in `sorted_unique_damages`
            # that is greater than or equal to `target_for_prev`.
            # Therefore, all elements before `prev_compatible_idx` (i.e., at indices 0 to `prev_compatible_idx - 1`)
            # are strictly less than `target_for_prev`.
            prev_compatible_idx = bisect.bisect_left(sorted_unique_damages, target_for_prev)
            
            prev_max_damage = 0
            # If `prev_compatible_idx` is greater than 0, it means there are elements
            # in `sorted_unique_damages` that are compatible (strictly less than `target_for_prev`).
            # The maximum damage from these compatible elements is stored at `dp[prev_compatible_idx - 1]`.
            if prev_compatible_idx > 0:
                prev_max_damage = dp[prev_compatible_idx - 1]
            
            # The total damage if we choose `current_damage_val` is its total damage plus
            # the maximum compatible damage from previous values.
            take_current = current_total_damage + prev_max_damage
            
            # The maximum damage for dp[i] is the maximum of these two options.
            dp[i] = max(take_none_current, take_current)
            
        # The final result is the maximum total damage considering all unique damage values,
        # which is stored in the last element of the dp array.
        return dp[n-1]