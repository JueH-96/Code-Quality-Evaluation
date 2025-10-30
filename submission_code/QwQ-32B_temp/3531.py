from typing import List
from functools import cmp_to_key

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        enemies = []
        for d, h in zip(damage, health):
            t = (h + power - 1) // power
            enemies.append((d, t))
        
        def compare(a, b):
            cross1 = a[0] * b[1]
            cross2 = b[0] * a[1]
            if cross1 > cross2:
                return -1
            elif cross1 < cross2:
                return 1
            else:
                return 0
        
        enemies.sort(key=cmp_to_key(compare))
        
        total_damage = 0
        current_sum = 0
        for d, t in enemies:
            current_sum += t
            total_damage += d * current_sum
        
        return total_damage