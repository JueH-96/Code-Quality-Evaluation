class Solution:
    def minDamage(self, power: int, damage: list[int], health: list[int]) -> int:
        # Calculate the number of seconds it takes to kill each enemy
        seconds_to_kill = [((h - 1) // power) + 1 for h in health]
        
        # Calculate the damage per second ratio for each enemy
        dps_ratio = [d / s for d, s in zip(damage, seconds_to_kill)]
        
        # Sort enemies by their damage per second ratio in descending order
        enemies = sorted(range(len(damage)), key=lambda i: dps_ratio[i], reverse=True)
        
        total_damage = 0
        current_dps = sum(damage)
        
        # Attack enemies in the order of their damage per second ratio
        for enemy in enemies:
            # Calculate the damage dealt to Bob while attacking this enemy
            total_damage += current_dps * seconds_to_kill[enemy]
            # Reduce the current damage per second after killing the enemy
            current_dps -= damage[enemy]
        
        return total_damage