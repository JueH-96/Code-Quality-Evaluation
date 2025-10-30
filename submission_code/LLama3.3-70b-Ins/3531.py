from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        # Calculate the total damage dealt by each enemy
        total_damage = [d * (h + power - 1) // power for d, h in zip(damage, health)]
        
        # Sort the enemies by the total damage they deal in descending order
        sorted_enemies = sorted(enumerate(total_damage), key=lambda x: x[1], reverse=True)
        
        # Initialize the total damage dealt to Bob
        total_damage_dealt = 0
        
        # Iterate over the sorted enemies
        for i, _ in sorted_enemies:
            # Calculate the number of seconds the enemy will be alive
            seconds_alive = (health[i] + power - 1) // power
            
            # Calculate the damage dealt by the enemy in the last second
            last_second_damage = damage[i]
            
            # Calculate the damage dealt by the enemy in the previous seconds
            previous_seconds_damage = damage[i] * (seconds_alive - 1)
            
            # Add the damage dealt by the enemy to the total damage dealt to Bob
            total_damage_dealt += previous_seconds_damage + last_second_damage
        
        return total_damage_dealt