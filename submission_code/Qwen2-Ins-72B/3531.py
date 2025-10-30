from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        total_damage = 0
        for i in range(n):
            # Calculate the number of seconds it takes to defeat the enemy
            seconds_to_defeat = (health[i] + power - 1) // power
            # Calculate the total damage dealt by the enemy
            total_damage += seconds_to_defeat * damage[i]
        return total_damage