class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        enemies = list(zip(damage, health))
        enemies.sort(key=lambda x: (-x[0], x[1]))
        current_time = 0
        total_damage = 0
        for d, h in enemies:
            k = (h + power - 1) // power
            current_time += k
            total_damage += d * current_time
        return total_damage