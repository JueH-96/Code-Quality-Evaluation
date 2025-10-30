from typing import List
import heapq

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        # Calculate the number of seconds required to kill each enemy
        time_to_kill = [(health[i] + power - 1) // power for i in range(len(health))]
        
        # Create a list of tuples (damage[i], time_to_kill[i]) for each enemy
        enemies = [(damage[i], time_to_kill[i]) for i in range(len(damage))]
        
        # Sort enemies by damage descending, then by time_to_kill ascending
        enemies.sort(key=lambda x: (-x[0], x[1]))
        
        total_damage = 0
        for damage_per_second, time in enemies:
            # For each enemy, they will deal damage for 'time' seconds
            total_damage += damage_per_second * time
        
        return total_damage