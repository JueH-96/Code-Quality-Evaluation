class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        import heapq
        
        # Create a list of enemies with their damage, health, and effective time to kill
        enemies = []
        for i in range(len(damage)):
            time_to_kill = (health[i] + power - 1) // power  # Ceiling division of health by power
            enemies.append((-damage[i], time_to_kill, damage[i]))  # Use negative damage for max-heap
        
        # Use a heap to always try to kill the enemy with the highest damage rate first
        heapq.heapify(enemies)
        
        total_damage = 0
        current_damage_per_second = sum(damage)
        
        while enemies:
            # Get the enemy with the highest damage rate
            neg_dmg, ttk, dmg = heapq.heappop(enemies)
            
            # Calculate the damage dealt to Bob while killing this enemy
            total_damage += current_damage_per_second * ttk
            
            # Reduce the damage per second by the damage of the killed enemy
            current_damage_per_second -= dmg
        
        return total_damage