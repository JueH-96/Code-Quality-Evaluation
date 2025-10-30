class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        enemies = []
        for i in range(n):
            enemies.append((damage[i], health[i]))
        
        enemies.sort(key=lambda x: x[0], reverse=True)
        
        total_damage = 0
        
        for enemy in enemies:
            d, h = enemy
            
            time_to_kill = (h + power - 1) // power
            
            total_damage += time_to_kill * sum(enemy[0] for enemy in enemies if enemy[1] > 0)
            
            h -= time_to_kill * power
            
            
            for i in range(len(enemies)):
                if enemies[i][0] == d and enemies[i][1] == enemy[1]:
                    enemies[i] = (d, h)
                    break
        
        return total_damage