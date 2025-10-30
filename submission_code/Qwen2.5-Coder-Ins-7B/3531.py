class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        enemies = sorted(zip(damage, health), key=lambda x: x[1] / x[0], reverse=True)
        total_damage = 0
        for d, h in enemies:
            if h > power:
                total_damage += power * (h // power + (1 if h % power != 0 else 0))
            else:
                total_damage += h
        return total_damage