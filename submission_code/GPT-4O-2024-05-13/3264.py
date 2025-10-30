class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        points = 0
        i, j = 0, len(enemyEnergies) - 1
        
        while i <= j:
            if currentEnergy >= enemyEnergies[i]:
                currentEnergy -= enemyEnergies[i]
                points += 1
                i += 1
            elif points > 0:
                currentEnergy += enemyEnergies[j]
                points -= 1
                j -= 1
            else:
                break
        
        return points