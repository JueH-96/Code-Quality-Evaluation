from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        enemies = []
        for d, h in zip(damage, health):
            k = (h + power - 1) // power
            enemies.append((d, k))
        
        # Sort enemies by descending order of damage per attack (d/k)
        enemies.sort(key=lambda x: -(x[0] / x[1]))
        
        total_damage = 0
        sum_k = 0
        for d, k in enemies:
            sum_k += k
            total_damage += d * sum_k
        
        return total_damage