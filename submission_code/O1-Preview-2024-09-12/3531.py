class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        enemies = []
        for i in range(n):
            # Sorting key: (-damage[i], health[i])
            enemies.append((-damage[i], health[i], damage[i]))
        enemies.sort()
        
        total_damage_per_second = sum(damage)
        total_damage = 0
        
        for _, h, d in enemies:
            time_needed = (h + power - 1) // power  # Ceiling division
            total_damage += total_damage_per_second * time_needed
            total_damage_per_second -= d  # Enemy is defeated, subtract its damage
        return total_damage