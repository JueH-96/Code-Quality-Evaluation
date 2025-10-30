class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        # Calculate time to kill each enemy
        times = [(health[i] + power - 1) // power for i in range(len(health))]
        
        # Create a list of (ratio, index) and sort by ratio in descending order
        enemies = [(damage[i] / times[i], i) for i in range(len(damage))]
        enemies.sort(reverse=True)
        
        total_damage = 0
        total_alive_damage = sum(damage)
        
        for ratio, i in enemies:
            # Add damage for the time it takes to kill this enemy
            total_damage += total_alive_damage * times[i]
            # Remove this enemy's damage from the total
            total_alive_damage -= damage[i]
        
        return total_damage