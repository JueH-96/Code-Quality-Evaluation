from typing import List
from functools import cmp_to_key

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        enemies = []
        n = len(damage)
        for i in range(n):
            k = (health[i] + power - 1) // power
            enemies.append((damage[i], k))
        
        # Define the comparator function for sorting
        def compare(a, b):
            diff = a[0] * b[1] - b[0] * a[1]
            if diff > 0:
                return -1
            elif diff < 0:
                return 1
            else:
                return 0
        
        # Sort enemies in descending order of damage/k ratio
        enemies.sort(key=cmp_to_key(compare))
        
        total_time = 0
        total_damage = 0
        for d, k in enemies:
            total_time += k
            total_damage += d * total_time
        
        return total_damage