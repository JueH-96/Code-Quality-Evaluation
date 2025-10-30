from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        points = 0
        i = 0
        n = len(enemyEnergies)
        
        while i < n:
            if currentEnergy >= enemyEnergies[i]:
                points += 1
                currentEnergy -= enemyEnergies[i]
                i += 1
            else:
                if points > 0:
                    currentEnergy += enemyEnergies[i]
                    i += 1
                else:
                    break
        
        return points