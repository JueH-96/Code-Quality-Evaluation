class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        enemies = []
        for i in range(n):
            enemies.append((damage[i], health[i], i))
        
        enemies.sort(reverse=True)
        
        total_damage = 0
        
        while any(h > 0 for _, h, _ in enemies):
            max_damage_enemy = -1
            max_damage = -1
            
            for i in range(len(enemies)):
                if enemies[i][1] > 0 and enemies[i][0] > max_damage:
                    max_damage = enemies[i][0]
                    max_damage_enemy = i

            turns = (enemies[max_damage_enemy][1] + power -1 ) // power
            current_damage = 0
            for i in range(len(enemies)):
                if enemies[i][1] > 0:
                    current_damage += enemies[i][0] * turns

            total_damage += current_damage
            enemies[max_damage_enemy] = (enemies[max_damage_enemy][0], max(0, enemies[max_damage_enemy][1] - power * turns), enemies[max_damage_enemy][2])

        return total_damage