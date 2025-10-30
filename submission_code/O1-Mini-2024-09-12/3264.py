from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        start, end = 0, len(enemyEnergies) - 1
        points = 0
        
        while start <= end:
            while start <= end and currentEnergy >= enemyEnergies[start]:
                currentEnergy -= enemyEnergies[start]
                points +=1
            if start > end:
                break
            if points >=1:
                currentEnergy += enemyEnergies[end]
                end -=1
            else:
                break
        return points