class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        points = 0
        marked_enemies = set()
        for i in range(len(enemyEnergies)):
            if currentEnergy >= enemyEnergies[i]:
                points += 1
                currentEnergy -= enemyEnergies[i]
            else:
                marked_enemies.add(i)
        while currentEnergy < len(enemyEnergies) and points > 0:
            points += 1
            currentEnergy += enemyEnergies[i]
            marked_enemies.add(i)
        return points