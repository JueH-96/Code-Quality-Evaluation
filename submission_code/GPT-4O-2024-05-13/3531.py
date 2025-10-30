class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        import heapq
        
        # Create a list of tuples (damage, health) for each enemy
        enemies = [(damage[i], health[i]) for i in range(len(damage))]
        
        # Sort enemies by damage per second in descending order
        enemies.sort(reverse=True, key=lambda x: x[0])
        
        total_damage = 0
        
        while enemies:
            # Get the enemy with the highest damage per second
            dmg, hp = enemies.pop(0)
            
            # Calculate the number of seconds needed to kill this enemy
            seconds_to_kill = (hp + power - 1) // power
            
            # Add the damage dealt by all remaining enemies during these seconds
            for i in range(len(enemies)):
                total_damage += enemies[i][0] * seconds_to_kill
            
            # Remove the killed enemy from the list
            enemies = [(d, h - seconds_to_kill * power) for d, h in enemies if h - seconds_to_kill * power > 0]
        
        return total_damage