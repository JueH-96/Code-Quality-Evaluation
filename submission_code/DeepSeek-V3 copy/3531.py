import heapq

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        # Create a list of tuples (damage, health) and sort in descending order of damage
        enemies = sorted(zip(damage, health), key=lambda x: -x[0])
        
        total_damage = 0
        current_damage = 0
        
        for d, h in enemies:
            # Calculate the number of seconds needed to kill this enemy
            seconds = (h + power - 1) // power
            # The total damage is the current damage multiplied by the seconds
            total_damage += current_damage * seconds
            # Update the current damage by adding the damage of this enemy
            current_damage += d
        
        return total_damage