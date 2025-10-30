from typing import List
import functools

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        enemies = []
        for d, h in zip(damage, health):
            k = (h + power - 1) // power
            enemies.append((d, k))
        
        def comparator(a, b):
            d1, k1 = a
            d2, k2 = b
            if d1 * k2 > d2 * k1:
                return -1
            elif d1 * k2 < d2 * k1:
                return 1
            else:
                return 0
        
        enemies_sorted = sorted(enemies, key=functools.cmp_to_key(comparator))
        
        total_damage = 0
        current_sum = 0
        for d, k in enemies_sorted:
            current_sum += k
            total_damage += d * current_sum
        
        return total_damage