from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        enemies = sorted([(damage[i], health[i]) for i in range(n)], key=lambda x: -x[0] * x[1])
        total_damage = 0
        for d, h in enemies:
            rounds = (h + power - 1) // power
            total_damage += rounds * d
            power = max(0, power - rounds + 1)
        return total_damage