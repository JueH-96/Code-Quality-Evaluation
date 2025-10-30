from typing import List
import heapq

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        # Calculate the time required to kill each enemy
        time_to_kill = [(health[i] + power - 1) // power for i in range(len(damage))]
        # Sort enemies by the time required to kill them
        sorted_enemies = sorted(zip(damage, time_to_kill), key=lambda x: x[1])
        
        total_damage = 0
        current_time = 0
        
        for d, t in sorted_enemies:
            # Damage dealt to Bob before this enemy is killed
            total_damage += d * current_time
            # Increment the current time by the time to kill this enemy
            current_time += t
        
        return total_damage