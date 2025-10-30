import heapq

class Solution:
    def minDamage(self, power: int, damage: list[int], health: list[int]) -> int:
        n = len(damage)
        enemies = [(-damage[i]*((health[i]+power-1)//power), damage[i]) for i in range(n)]
        heapq.heapify(enemies)
        total_damage = 0
        while enemies:
            max_dmg, dmg = heapq.heappop(enemies)
            total_damage -= max_dmg
            if enemies:
                total_damage += enemies[0][1]*(((-max_dmg)//dmg)-1)
        return total_damage