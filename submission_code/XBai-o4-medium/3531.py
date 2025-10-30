from typing import List
from functools import cmp_to_key

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(health)
        enemies = []
        for d, h in zip(damage, health):
            t = (h + power - 1) // power
            enemies.append((d, t))
        
        def compare(a, b):
            val1 = a[0] * b[1]
            val2 = b[0] * a[1]
            if val1 > val2:
                return -1
            elif val1 < val2:
                return 1
            else:
                return 0
        
        enemies.sort(key=cmp_to_key(compare))
        
        suffix_sum = [0] * n
        suffix_sum[-1] = enemies[-1][0]
        for i in range(n - 2, -1, -1):
            suffix_sum[i] = enemies[i][0] + suffix_sum[i + 1]
        
        total = 0
        for i in range(n):
            total += enemies[i][1] * suffix_sum[i]
        
        return total