class Solution:
    def maximumPoints(self, enemyEnergies: list[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        points = 0
        i = 0
        while i < len(enemyEnergies):
            if currentEnergy >= enemyEnergies[i]:
                points += 1
                currentEnergy -= enemyEnergies[i]
                i += 1
            elif points > 0:
                points -= 1
                currentEnergy += enemyEnergies[i]
                enemyEnergies.pop(i)
            else:
                break
        return points