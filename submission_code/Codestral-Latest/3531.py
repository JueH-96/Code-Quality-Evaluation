from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        # Create a list of tuples where each tuple contains (damage, health)
        enemies = list(zip(damage, health))

        # Sort enemies by the damage they deal per second
        enemies.sort(key=lambda x: x[0])

        total_damage = 0

        for dmg, hp in enemies:
            # Calculate the number of seconds needed to kill the enemy
            seconds_needed = (hp + power - 1) // power
            # Calculate the damage dealt to Bob in that time
            total_damage += dmg * seconds_needed

        return total_damage