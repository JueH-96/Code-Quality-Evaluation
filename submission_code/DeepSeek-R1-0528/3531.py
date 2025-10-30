class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        enemies = []
        for i in range(len(damage)):
            t = (health[i] + power - 1) // power
            enemies.append((damage[i], t))
        
        enemies.sort(key=lambda x: x[0] / x[1], reverse=True)
        
        total_damage = 0
        cumulative_time = 0
        for d, t in enemies:
            cumulative_time += t
            total_damage += d * cumulative_time
        
        return total_damage