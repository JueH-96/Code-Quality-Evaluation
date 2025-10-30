class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        import math
        
        n = len(damage)
        
        # Calculate time to kill each enemy
        time_to_kill = []
        for i in range(n):
            time_to_kill.append(math.ceil(health[i] / power))
        
        # Create list of (index, damage_per_time) and sort by damage_per_time descending
        enemies = []
        for i in range(n):
            # We want to maximize damage[i] / time_to_kill[i]
            # To avoid floating point, we can compare damage[i] * time_to_kill[j] vs damage[j] * time_to_kill[i]
            enemies.append(i)
        
        # Sort enemies by damage/time ratio in descending order
        # For enemies i and j, i should come before j if damage[i]/time[i] > damage[j]/time[j]
        # Which is equivalent to damage[i] * time[j] > damage[j] * time[i]
        enemies.sort(key=lambda i: -damage[i] / time_to_kill[i])
        
        # Calculate total damage
        total_damage = 0
        current_damage_per_second = sum(damage)
        
        for enemy_idx in enemies:
            # While killing this enemy, we take damage from all alive enemies
            total_damage += current_damage_per_second * time_to_kill[enemy_idx]
            # After killing this enemy, reduce the damage per second
            current_damage_per_second -= damage[enemy_idx]
        
        return total_damage