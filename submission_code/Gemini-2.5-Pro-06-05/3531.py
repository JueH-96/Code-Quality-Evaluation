from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        from fractions import Fraction

        n = len(damage)
        
        # Combine enemy data into a single list of dictionaries for easier handling.
        enemies = []
        for i in range(n):
            d = damage[i]
            h = health[i]
            
            # Calculate the number of hits (seconds) required to defeat this enemy.
            # This is equivalent to ceil(health / power).
            time_to_kill = (h + power - 1) // power
            
            # Based on constraints (health >= 1, power >= 1), time_to_kill is always >= 1.
            # This avoids potential division by zero later.
            enemies.append({'damage': d, 'time': time_to_kill})

        # The core insight is that this is a weighted completion time scheduling problem.
        # The optimal strategy is to defeat enemies in descending order of their
        # "damage / time_to_kill" ratio (Smith's Rule).
        # We use `fractions.Fraction` to ensure precise sorting and avoid floating-point errors.
        enemies.sort(key=lambda e: Fraction(e['damage'], e['time']), reverse=True)
        
        # Calculate the total damage by simulating the fight with the optimal kill order.
        total_damage_taken = 0
        
        # The initial damage per second is the sum of damage from all enemies.
        current_total_dps = sum(damage)
        
        # Iterate through the sorted enemies, defeating them one by one.
        for enemy in enemies:
            # While defeating the current enemy, all enemies that are still alive deal damage.
            # The damage incurred during this phase is time_to_kill * current_total_dps.
            total_damage_taken += enemy['time'] * current_total_dps
            
            # Once this enemy is defeated, it no longer contributes to the damage per second.
            current_total_dps -= enemy['damage']
            
        return total_damage_taken