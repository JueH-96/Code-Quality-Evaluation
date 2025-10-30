import math
from functools import cmp_to_key
from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        
        # Create list of enemies as (damage, attacks_needed) tuples
        enemies = []
        for i in range(n):
            # Calculate attacks needed for enemy i: ceil(health[i] / power)
            attacks_needed = (health[i] + power - 1) // power
            enemies.append((damage[i], attacks_needed))
            
        # Sort enemies based on the optimal criterion.
        # Consider two enemies i and j with (d_i, a_i) and (d_j, a_j).
        # If enemy i is defeated immediately before enemy j, the additional damage
        # incurred from enemy j while attacking enemy i is d_j * a_i.
        # If enemy j is defeated immediately before enemy i, the additional damage
        # incurred from enemy i while attacking enemy j is d_i * a_j.
        # To minimize damage, we prefer the order (i then j) if d_j * a_i < d_i * a_j.
        # This means we want to sort such that if enemy i comes before enemy j in the sorted list,
        # then d_j * a_i >= d_i * a_j.
        # The comparison function for cmp_to_key(lambda item1, item2: cmp_val) should return
        # a negative value if item1 is considered "less than" item2 (comes first in ascending sort).
        # We want item1=(d_i, a_i) to come before item2=(d_j, a_j) if d_j * a_i < d_i * a_j.
        # The value d_j * a_i - d_i * a_j is negative exactly when d_j * a_i < d_i * a_j.
        # So the comparison function is lambda item1, item2: item2[0] * item1[1] - item1[0] * item2[1].
        enemies.sort(key=cmp_to_key(lambda item1, item2: item2[0] * item1[1] - item1[0] * item2[1]))
        
        # Calculate total damage
        # Iterate through the sorted enemies. For each enemy k in the sorted order (index p_k),
        # it takes A_{p_k} seconds to defeat it. During these A_{p_k} seconds,
        # all enemies from index k onwards in the sorted list (p_k, p_{k+1}, ..., p_{n-1})
        # are still alive and dealing damage.
        # The total damage received is sum across k = 0 to n-1 of:
        # (sum of damage of enemies p_k, ..., p_{n-1}) * A_{p_k}
        total_damage = 0
        
        # Calculate initial total damage per second from all enemies
        current_total_damage_per_sec = sum(d for d, a in enemies)
        
        # Iterate through enemies in the sorted order
        for d, a in enemies:
            # While attacking the current enemy (which needs 'a' seconds),
            # the total damage received per second is current_total_damage_per_sec.
            total_damage += current_total_damage_per_sec * a
            
            # After defeating the current enemy, subtract its damage from the total
            # damage per second for the next phases.
            current_total_damage_per_sec -= d
            
        return total_damage