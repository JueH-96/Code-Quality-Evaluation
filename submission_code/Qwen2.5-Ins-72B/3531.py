from typing import List
import math

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        total_damage = 0
        n = len(damage)
        
        # Sort enemies by the number of seconds it takes to kill them
        enemies = sorted([(math.ceil(health[i] / power), damage[i]) for i in range(n)])
        
        for seconds, dmg in enemies:
            total_damage += seconds * sum(damage)
            damage.remove(dmg)
        
        return total_damage