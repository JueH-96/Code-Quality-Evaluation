import math
from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        # Compute the number of hits needed for each enemy
        hits = [(health[i] + power - 1) // power for i in range(n)]
        # Create a list of indices
        indices = list(range(n))
        # Sort indices in decreasing order of damage[i] / hits[i]
        indices.sort(key=lambda i: damage[i] / hits[i], reverse=True)
        # Compute the initial sum of all damages
        current_sum_damage = sum(damage)
        total_damage = 0
        # Iterate through the enemies in the sorted order
        for idx in indices:
            total_damage += hits[idx] * current_sum_damage
            current_sum_damage -= damage[idx]
        return total_damage