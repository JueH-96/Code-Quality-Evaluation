from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        enemies = []
        for d, h in zip(damage, health):
            k = (h + power - 1) // power
            enemies.append((d, k))
        
        # Sort enemies by the ratio damage/k in descending order
        enemies.sort(key=lambda x: (-(x[0] / x[1]), x[0], x[1]))
        
        total_damage = 0
        sum_k = 0
        for d, k in enemies:
            total_damage += (sum_k + k) * d
            sum_k += k
        
        return total_damage