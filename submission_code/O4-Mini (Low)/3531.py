from typing import List
from functools import cmp_to_key

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        # Compute processing time (number of seconds) for each enemy
        times = [(health[i] + power - 1) // power for i in range(n)]
        
        # Prepare list of (weight, time) pairs
        jobs = [(damage[i], times[i]) for i in range(n)]
        
        # Compare function to sort by decreasing weight/time ratio
        def cmp(a, b):
            w1, t1 = a
            w2, t2 = b
            # if w1/t1 > w2/t2 => w1 * t2 > w2 * t1
            if w1 * t2 > w2 * t1:
                return -1
            elif w1 * t2 < w2 * t1:
                return 1
            else:
                return 0
        
        # Sort jobs by ratio w/t descending
        jobs.sort(key=cmp_to_key(cmp))
        
        total_damage = 0
        elapsed = 0
        # Accumulate weighted completion times
        for w, t in jobs:
            elapsed += t
            total_damage += w * elapsed
        
        return total_damage