from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        points = 0
        marked = []
        
        for energy in enemyEnergies:
            if currentEnergy >= energy:
                points += 1
                currentEnergy -= energy
                marked.append(energy)
            elif points > 0 and marked:
                currentEnergy += marked.pop()
                currentEnergy -= energy
                marked.append(energy)
        
        return points