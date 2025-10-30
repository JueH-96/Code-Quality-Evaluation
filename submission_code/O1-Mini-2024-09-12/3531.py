from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        enemies = []
        total_dps = 0
        n = len(damage)
        for d, h in zip(damage, health):
            process_time = (h + power - 1) // power
            enemies.append((d, process_time))
            total_dps += d
        # Sort enemies by descending order of damage[i] / process_time[i]
        enemies.sort(key=lambda x: -x[0] / x[1])
        total_damage = 0
        for d, t in enemies:
            total_damage += total_dps * t
            total_dps -= d
        return total_damage