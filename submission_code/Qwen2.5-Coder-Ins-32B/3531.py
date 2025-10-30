from typing import List
import math

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        total_damage = 0
        for d, h in zip(damage, health):
            # Calculate the number of seconds needed to kill the enemy
            seconds_to_kill = math.ceil(h / power)
            # Calculate the total damage Bob takes from this enemy
            total_damage += d * seconds_to_kill
        return total_damage