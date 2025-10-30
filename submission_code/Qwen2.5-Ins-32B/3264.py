from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        points = 0
        i = 0
        while i < len(enemyEnergies) and currentEnergy >= enemyEnergies[i]:
            currentEnergy -= enemyEnergies[i]
            points += 1
            i += 1
        return points + currentEnergy // sum(enemyEnergies[i:])