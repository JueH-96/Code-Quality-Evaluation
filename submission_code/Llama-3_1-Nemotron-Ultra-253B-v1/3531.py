from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        enemies = []
        for d, h in zip(damage, health):
            t = (h + power - 1) // power
            enemies.append((d, t))
        
        # Sort enemies by descending damage per time ratio
        enemies.sort(key=lambda x: -x[0] / x[1])
        
        total_damage = 0
        sum_t = 0
        for d, t in enemies:
            total_damage += d * (sum_t + t)
            sum_t += t
        
        return total_damage