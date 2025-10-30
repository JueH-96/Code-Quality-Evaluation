class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        import math
        
        n = len(damage)
        
        # Calculate time to kill each enemy and create list of (efficiency, index)
        enemies = []
        for i in range(n):
            time_to_kill = math.ceil(health[i] / power)
            efficiency = damage[i] / time_to_kill
            enemies.append((efficiency, i, time_to_kill))
        
        # Sort by efficiency in descending order
        enemies.sort(reverse=True)
        
        total_damage = 0
        current_time = 0
        total_dps = sum(damage)  # Total damage per second from all alive enemies
        
        # Kill enemies in order of efficiency
        for efficiency, enemy_idx, time_to_kill in enemies:
            # For the time it takes to kill this enemy, take damage from all current alive enemies
            total_damage += total_dps * time_to_kill
            current_time += time_to_kill
            
            # Remove this enemy's damage from future calculations
            total_dps -= damage[enemy_idx]
        
        return total_damage