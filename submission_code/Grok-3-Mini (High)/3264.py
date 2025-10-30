from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        total_sum = sum(enemyEnergies)
        candidates = set()
        for energy in enemyEnergies:
            if energy <= currentEnergy:
                candidates.add(energy)
        if not candidates:
            return 0
        max_points = 0
        for C in candidates:
            energy_for_defeating = currentEnergy + total_sum - C
            points = energy_for_defeating // C
            if points > max_points:
                max_points = points
        return max_points