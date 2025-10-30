import math

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        time_to_kill = []
        ratios = []
        indices = list(range(n))
        for i in range(n):
            t = math.ceil(health[i] / power)
            time_to_kill.append(t)
            ratios.append(damage[i] / t if t > 0 else 0)
        
        enemy_order_indices = sorted(indices, key=lambda i: ratios[i], reverse=True)
        
        current_health = list(health)
        current_damage = list(damage)
        total_damage_taken = 0
        
        for enemy_index in enemy_order_indices:
            if current_health[enemy_index] <= 0:
                continue
            
            damage_per_second = 0
            for i in range(n):
                if current_health[i] > 0:
                    damage_per_second += current_damage[i]
                    
            seconds_to_kill = math.ceil(current_health[enemy_index] / power)
            damage_taken_in_interval = damage_per_second * seconds_to_kill
            total_damage_taken += damage_taken_in_interval
            current_health[enemy_index] = 0
            
        return total_damage_taken