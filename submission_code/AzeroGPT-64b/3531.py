from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        # Sort the health array along with damage array based on health.
        # This is to ensure that the enemies with the least health are targeted first.
        # This approach is justified because the higher the health of an enemy,
        # the more total damage they would inflict on Bob (given that the damage per second of enemies is non-decreasing).
        # Therefore, eliminating enemies with lower health earlier reduces total damage to Bob.
        sorted_enemies = sorted(zip(damage, health), key=lambda x: x[1])
        
        # Initialize variables to track the total damage and the time spent.
        total_damage = 0
        time = 0
        
        for enemy_damage, enemy_health in sorted_enemies:
            # Calculate the time needed to defeat this enemy.
            time_to_defeat = -(-enemy_health // power)  # Ceiling division
            
            # Update the total damage based on the time the enemy was alive.
            total_damage += time_to_defeat * enemy_damage
            
            # Update the time by the time taken to defeat this enemy.
            time += time_to_defeat
        
        return total_damage