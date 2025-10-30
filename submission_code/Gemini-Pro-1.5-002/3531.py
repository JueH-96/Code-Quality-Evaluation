class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        enemies = []
        for i in range(n):
            enemies.append((health[i], damage[i]))
        enemies.sort(key=lambda x: x[1] / x[0], reverse=True)
        
        total_damage = 0
        for i in range(n):
            kills = (enemies[i][0] + power - 1) // power
            
            damage_sum = 0
            for j in range(n):
                if j <= i:
                    damage_sum += enemies[j][1] * kills
                else:
                    damage_sum += enemies[j][1] * ((enemies[i][0] + power - 1) // power)
            
            total_damage += damage_sum - enemies[i][1] * kills
            
            for j in range(i + 1, n):
                total_damage += enemies[j][1] * ((enemies[i][0] + power - 1) // power - (enemies[i][0] - 1) // power)


        return total_damage