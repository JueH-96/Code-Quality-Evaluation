from typing import List
import math

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        # For each enemy i, compute the number of rounds (hits) required to kill it.
        # rounds = ceil(health[i] / power)
        # Each enemy i, if killed after C rounds, will contribute its damage[i] for C seconds.
        # So the total damage Bob takes is sum(damage[i] * C_i) over all enemies.
        # This is equivalent to the scheduling problem of minimizing the weighted sum of completion times,
        # where each enemy is a job with processing time = rounds and weight = damage.
        # The optimal policy is to process jobs in increasing order of (rounds / damage).
        enemies = []
        for i in range(n):
            rounds = (health[i] + power - 1) // power  # ceil division: rounds to kill enemy i
            enemies.append((rounds, damage[i]))
        
        # Sort enemies by increasing ratio rounds/damage, i.e. by rounds/damage value.
        enemies.sort(key=lambda x: x[0] / x[1])
        
        total_damage = 0
        cumulative_rounds = 0
        # In the sorted order, each enemy's death time is the sum of rounds for it and all previously killed enemies.
        for rounds, d in enemies:
            cumulative_rounds += rounds
            total_damage += cumulative_rounds * d
        
        return total_damage