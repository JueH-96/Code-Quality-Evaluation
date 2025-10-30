class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        points = 0
        n = len(enemyEnergies)
        for i in range(n):
            if currentEnergy >= enemyEnergies[i]:
                currentEnergy -= enemyEnergies[i]
                points += 1
            else:
                if points > 0:
                    currentEnergy += enemyEnergies[n - 1]
                    n -= 1
                    if currentEnergy >= enemyEnergies[i]:
                        currentEnergy -= enemyEnergies[i]
                        points += 1
                    else:
                        break
                else:
                    break

        return points