import math
from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)

        # Step 1: Calculate hits needed for each enemy and store their (damage, hits_needed)
        # tuples. Also, calculate the initial total damage per second dealt by all enemies.
        enemies_info = []
        current_total_damage_per_sec = 0
        for i in range(n):
            # Calculate hits_needed: ceil(health[i] / power)
            # This is equivalent to (health[i] + power - 1) // power using integer division.
            hits_needed = (health[i] + power - 1) // power
            enemies_info.append((damage[i], hits_needed))
            current_total_damage_per_sec += damage[i]

        # Step 2: Sort enemies based on the greedy strategy.
        # Prioritize enemies with a higher (damage / hits_needed) ratio.
        # In case of a tie in ratio, the order does not affect the total minimum damage.
        # However, for consistency and potentially minor performance benefits (removing damage source quicker),
        # we can tie-break by prioritizing enemies that require fewer hits (smaller hits_needed).
        #
        # The sorting key `lambda x: (-x[0] / x[1], x[1])` achieves this:
        # - `-x[0] / x[1]` (negative of damage / hits_needed) sorts primarily by ratio in descending order.
        # - `x[1]` (hits_needed) sorts secondarily by hits_needed in ascending order for ties.
        enemies_info.sort(key=lambda x: (-x[0] / x[1], x[1]))

        total_damage_to_bob = 0
        
        # Step 3: Simulate the combat and calculate the total damage dealt to Bob.
        # We iterate through the enemies in the sorted order.
        for d, t in enemies_info:
            # While attacking the current enemy (which takes 't' seconds), all
            # currently alive enemies (including the one being attacked) continue to deal damage.
            # So, for 't' seconds, Bob takes 'current_total_damage_per_sec' damage per second.
            total_damage_to_bob += current_total_damage_per_sec * t
            
            # After 't' seconds, this enemy is defeated.
            # Its damage contribution is removed from the total for all subsequent seconds.
            current_total_damage_per_sec -= d
            
        return total_damage_to_bob