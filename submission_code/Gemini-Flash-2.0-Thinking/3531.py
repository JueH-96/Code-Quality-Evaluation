from typing import List
import math

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        enemies = sorted(range(n), key=lambda i: damage[i] / math.ceil(health[i] / power) if math.ceil(health[i] / power) > 0 else float('inf'), reverse=True)

        total_damage = 0
        current_health = list(health)

        for enemy_index in enemies:
            if current_health[enemy_index] > 0:
                time_to_kill = math.ceil(current_health[enemy_index] / power)
                current_damage_per_second = sum(damage[i] for i in range(n) if current_health[i] > 0)
                total_damage += current_damage_per_second * time_to_kill

                current_health[enemy_index] -= power * time_to_kill

        return total_damage