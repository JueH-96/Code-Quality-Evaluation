from typing import List
import heapq

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        total_damage = 0
        enemies = []
        
        # Create a max-heap based on the damage per second
        for i in range(n):
            # We store (-damage[i], health[i]) to create a max-heap
            heapq.heappush(enemies, (-damage[i], health[i]))

        while enemies:
            # Get the enemy with the highest damage
            current_damage, current_health = heapq.heappop(enemies)
            current_damage = -current_damage
            
            # Calculate how many seconds it takes to defeat this enemy
            seconds_to_defeat = (current_health + power - 1) // power
            
            # Calculate the damage dealt to Bob during these seconds
            total_damage += seconds_to_defeat * current_damage
            
            # Now we need to update the remaining enemies
            # We need to account for the time that passes while we defeat this enemy
            for i in range(len(enemies)):
                # Update the health of remaining enemies
                enemies[i] = (enemies[i][0], enemies[i][1] - seconds_to_defeat * enemies[i][0] * -1)
            
            # Clean up the heap to remove any defeated enemies
            enemies = [(d, h) for d, h in enemies if h > 0]
            heapq.heapify(enemies)

        return total_damage