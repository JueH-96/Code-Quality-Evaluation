class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        n = len(enemyEnergies)
        points = 0
        marked = set()
        
        for i in range(n):
            if currentEnergy >= enemyEnergies[i]:
                points += 1
                currentEnergy -= enemyEnergies[i]
                marked.add(i)
            elif points > 0:
                currentEnergy += enemyEnergies[i]
                marked.add(i)
        
        return points