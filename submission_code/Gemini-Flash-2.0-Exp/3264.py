class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        points = 0
        marked = set()
        i = 0
        while i < len(enemyEnergies):
            if currentEnergy >= enemyEnergies[i]:
                currentEnergy -= enemyEnergies[i]
                points += 1
                i += 1
            elif points > 0:
                currentEnergy += enemyEnergies[i]
                marked.add(i)
                i += 1
            else:
                break
        return points