from typing import List
from collections import Counter
from bisect import bisect_right

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        # Count total occurrences of each damage value.
        damage_counts = Counter(power)
        
        # For each unique damage value, compute the total damage if spells of that value are cast.
        # (Since spells with the same damage value do not conflict with each other, we take them all.)
        unique_damages = sorted(damage_counts.keys())
        total_damage_for_value = {d: d * damage_counts[d] for d in unique_damages}
        
        n = len(unique_damages)
        dp = [0] * n  # dp[i] will hold the maximum total damage using values up to unique_damages[i].
        
        for i, d in enumerate(unique_damages):
            current_value_total = total_damage_for_value[d]
            # Find the index of the largest damage value that does not conflict.
            # d conflicts with any damage within d-2, d-1, d+1, and d+2.
            # When processing in increasing order, we need a damage value <= d-3.
            # Using binary search to find the rightmost index j such that unique_damages[j] <= d - 3.
            j = bisect_right(unique_damages, d - 3) - 1
            if j >= 0:
                candidate = dp[j] + current_value_total
            else:
                candidate = current_value_total
            # Either we skip the current damage value or take it (and add the best safe solution before it).
            dp[i] = candidate if i == 0 else max(dp[i-1], candidate)
        
        return dp[-1]