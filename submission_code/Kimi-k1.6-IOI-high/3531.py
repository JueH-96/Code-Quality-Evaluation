from functools import cmp_to_key
from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        enemies = []
        for d, h in zip(damage, health):
            # Calculate the time needed to kill this enemy (ceil division)
            t = (h + power - 1) // power
            enemies.append((d, t))
        
        # Comparator function to sort enemies by descending order of (d / t)
        def compare(a, b):
            d1, t1 = a
            d2, t2 = b
            if d1 * t2 > d2 * t1:
                return -1
            elif d1 * t2 < d2 * t1:
                return 1
            else:
                return 0
        
        # Sort enemies using the custom comparator
        enemies.sort(key=cmp_to_key(compare))
        
        total_damage = 0
        time_sum = 0
        for d, t in enemies:
            time_sum += t
            total_damage += d * time_sum
        
        return total_damage