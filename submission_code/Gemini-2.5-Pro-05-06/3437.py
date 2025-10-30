import collections
import bisect
from typing import List

class Solution:
  def maximumTotalDamage(self, power: List[int]) -> int:
    # 1. Count frequencies of spell damages
    counts = collections.Counter(power)
    
    # 2. Get sorted unique damage values
    # unique_damages refers to sorted unique spell power values
    unique_damages = sorted(counts.keys())
    
    num_unique_damages = len(unique_damages)
    
    # dp[i] will store the maximum damage considering spells from unique_damages[0...i].
    # That is, dp[i] is the max damage from a subset of {unique_damages[0], ..., unique_damages[i]}
    # respecting the spell casting constraints.
    dp = [0] * num_unique_damages
    
    # 3. Fill DP table
    for j in range(num_unique_damages):
      current_spell_power = unique_damages[j]
      
      # Calculate the total damage obtained if we decide to cast all spells of this specific power value
      damage_from_current_power_spells = current_spell_power * counts[current_spell_power]
      
      # Option 1: Don't cast spells of power unique_damages[j].
      # In this case, the maximum damage is whatever we achieved considering spells up to unique_damages[j-1].
      # If j=0, there's no unique_damages[j-1], so this damage is 0.
      max_damage_if_skip_current = 0
      if j > 0:
        max_damage_if_skip_current = dp[j-1]
      
      # Option 2: Cast spells of power unique_damages[j].
      # The damage from these spells is damage_from_current_power_spells.
      # We must add to this the maximum damage from previous spells that are compatible.
      # A previous spell unique_damages[k] is compatible if unique_damages[k] < current_spell_power - 2.
      # The maximum damage accumulated considering spells up to unique_damages[k] is dp[k].
      max_damage_if_take_current = damage_from_current_power_spells
      
      # We need to find the largest index k such that unique_damages[k] < (current_spell_power - 2).
      # Let prev_spell_power_threshold = current_spell_power - 2.
      # We are looking for an index prev_compatible_dp_idx such that unique_damages[prev_compatible_dp_idx] < prev_spell_power_threshold.
      # bisect_left(arr, x) returns an index `ins_idx` such that for all i < ins_idx, arr[i] < x,
      # and for all i >= ins_idx, arr[i] >= x.
      # So, the elements at indices 0 to `ins_idx - 1` are < x.
      # The largest index for an element < x is `ins_idx - 1`. This index corresponds to dp array.
      prev_spell_power_threshold = current_spell_power - 2
      
      # ins_idx is the insertion point for `prev_spell_power_threshold` in `unique_damages`.
      # We are searching in the entire `unique_damages` array.
      # `prev_spell_power_threshold` is strictly less than `current_spell_power = unique_damages[j]`
      # (Unless current_spell_power is 1 or 2, then threshold can be <=0, in which case ins_idx will be 0).
      # So `bisect.bisect_left(unique_damages, prev_spell_power_threshold)` will return an index `<= j`.
      ins_idx = bisect.bisect_left(unique_damages, prev_spell_power_threshold)
      # prev_compatible_dp_idx is the index in unique_damages (and thus dp) of the latest spell type
      # that is compatible with current_spell_power.
      prev_compatible_dp_idx = ins_idx - 1
      
      if prev_compatible_dp_idx >= 0:
        max_damage_if_take_current += dp[prev_compatible_dp_idx]
      
      # dp[j] is the maximum of these two options.
      dp[j] = max(max_damage_if_skip_current, max_damage_if_take_current)
            
    # The final answer is dp[num_unique_damages - 1].
    # This is because dp[i] stores the max damage using a subset of the first i+1 unique spell powers.
    # Thus dp[num_unique_damages - 1] considers all available unique spell powers.
    # Constraints state power.length >= 1, so num_unique_damages >= 1.
    # Thus, num_unique_damages - 1 is a valid index (>=0).
    return dp[num_unique_damages - 1]