import math

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        enemies = []
        for i in range(n):
            enemies.append({'damage': damage[i], 'health': health[i], 'index': i})
        enemies.sort(key=lambda x: x['damage'], reverse=True)
        current_health = list(health)
        alive_enemy_indices = set(range(n))
        total_damage_taken = 0
        for enemy_info in enemies:
            enemy_index = enemy_info['index']
            if enemy_index not in alive_enemy_indices:
                continue
            current_damage_per_second = 0
            for index in alive_enemy_indices:
                current_damage_per_second += damage[index]
            seconds_to_kill = math.ceil(current_health[enemy_index] / power)
            damage_taken_in_seconds = current_damage_per_second * seconds_to_kill
            total_damage_taken += damage_taken_in_seconds
            current_health[enemy_index] -= power * seconds_to_kill
            if current_health[enemy_index] <= 0:
                alive_enemy_indices.remove(enemy_index)
        return total_damage_taken