class Solution:
    def minDamage(self, power: int, damage: list[int], health: list[int]) -> int:
        n = len(damage)
        enemies = []
        for i in range(n):
            enemies.append((damage[i], health[i]))

        enemies.sort(reverse=True)

        total_damage = 0
        while any(h > 0 for _, h in enemies):
            current_damage = sum(d for d, h in enemies if h > 0)
            total_damage += current_damage

            max_damage_index = -1
            max_damage = -1

            for i in range(n):
                if enemies[i][1] > 0:
                    if enemies[i][0] > max_damage:
                        max_damage = enemies[i][0]
                        max_damage_index = i

            if max_damage_index != -1:
                enemies[max_damage_index] = (enemies[max_damage_index][0], enemies[max_damage_index][1] - power)

        return total_damage