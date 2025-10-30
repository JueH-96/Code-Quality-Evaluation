from typing import List
from functools import cmp_to_key

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        enemies = []
        for d, h in zip(damage, health):
            k = (h + power - 1) // power
            enemies.append((d, k))
        
        def compare(a, b):
            d1, k1 = a
            d2, k2 = b
            val = d1 * k2 - d2 * k1
            if val > 0:
                return -1
            elif val < 0:
                return 1
            else:
                return 0
        
        enemies.sort(key=cmp_to_key(compare))
        
        total_time = 0
        total_damage = 0
        for d, k in enemies:
            total_time += k
            total_damage += d * total_time
        
        return total_damage