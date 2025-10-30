import functools
from typing import List

class Solution:
  def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
    n = len(damage)
    enemies_info = []
    # Store relevant info for each enemy: time to kill, damage per second, original index
    for i in range(n):
      h = health[i]
      d = damage[i]
      # Calculate time to kill enemy i: ceil(health[i] / power)
      # For positive integers A, B, ceil(A/B) can be calculated as (A + B - 1) // B
      t = (h + power - 1) // power
      enemies_info.append({'t': t, 'd': d, 'original_idx': i})

    # Custom comparison function for sorting enemies.
    # The core idea is to sort by the ratio t_k / d_k in ascending order.
    # An enemy e1 comes before e2 if e1.t / e1.d < e2.t / e2.d.
    # To avoid floating point arithmetic, this is equivalent to e1.t * e2.d < e2.t * e1.d,
    # since d (damage values) are positive according to constraints (damage[i] >= 1).
    def compare_enemies(e1, e2):
      # Primary sort key: ratio t/d
      # Compare e1['t']*e2['d'] vs e2['t']*e1['d']
      
      val_e1_ratio_metric = e1['t'] * e2['d']
      val_e2_ratio_metric = e2['t'] * e1['d']

      if val_e1_ratio_metric < val_e2_ratio_metric:
        return -1 # e1 comes first
      if val_e1_ratio_metric > val_e2_ratio_metric:
        return 1  # e2 comes first
      
      # Ratios t/d are equal if val_e1_ratio_metric == val_e2_ratio_metric.
      # Apply tie-breaking rules. A common tie-breaker for WSPT is to prioritize
      # jobs with higher weight (larger damage[k]) first.
      if e1['d'] > e2['d']:
        return -1 # e1 comes first (higher damage implies it's "heavier weight")
      if e1['d'] < e2['d']:
        return 1  # e2 comes first
      
      # Damages are equal. Since ratios t/d are also equal, times t must be equal too.
      # (e.g., if t1/d_common = t2/d_common, then t1 must be equal to t2).
      # As a final tie-breaker for full stability (though not strictly necessary for sum),
      # use original index. Smaller index first.
      if e1['original_idx'] < e2['original_idx']:
        return -1
      if e1['original_idx'] > e2['original_idx']:
        return 1
      
      return 0 # Should only be reached if it's effectively the same element, 
               # which compare_enemies wouldn't see due to distinct original_idx.

    # Sort enemies based on the custom comparison logic
    enemies_info.sort(key=functools.cmp_to_key(compare_enemies))
    
    total_damage_taken = 0
    
    # Calculate initial sum of DPS from all enemies
    current_total_dps = 0
    for d_val in damage: # sum of all damage[i]
        current_total_dps += d_val

    # Iterate through the sorted enemies, simulating the process of killing them one by one
    for enemy_stats in enemies_info:
      t_k = enemy_stats['t'] # Time Bob spends attacking this enemy until it's dead
      d_k = enemy_stats['d'] # DPS of this specific enemy
      
      # While Bob is attacking this enemy (for t_k seconds), all currently alive enemies
      # (including this one) continue to deal damage.
      # The damage taken during these t_k seconds is current_total_dps * t_k.
      total_damage_taken += current_total_dps * t_k
      
      # After t_k seconds, this enemy is dead. Remove its DPS contribution for future calculations.
      current_total_dps -= d_k
            
    return total_damage_taken