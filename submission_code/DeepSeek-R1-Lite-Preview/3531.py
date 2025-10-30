class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        import math
        n = len(damage)
        enemies = []
        for i in range(n):
            p_i = (health[i] + power - 1) // power  # ceil(health[i]/power)
            enemies.append((damage[i], p_i))
        
        # Sort enemies by damage[i]/p_i in decreasing order
        enemies.sort(key=lambda x: -x[0]/x[1])
        
        total_damage = 0
        cumulative_p = 0
        for d, p in enemies:
            cumulative_p += p
            total_damage += d * cumulative_p
        
        return total_damage