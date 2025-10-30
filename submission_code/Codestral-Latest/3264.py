class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        # Sort the enemy energies in ascending order
        enemyEnergies.sort()

        points = 0
        marked_enemies = set()

        # First, try to gain points by reducing currentEnergy
        for i in range(len(enemyEnergies)):
            if currentEnergy >= enemyEnergies[i]:
                currentEnergy -= enemyEnergies[i]
                points += 1
                marked_enemies.add(i)

        # Then, try to gain energy back by marking enemies
        for i in range(len(enemyEnergies)):
            if i not in marked_enemies and points > 0:
                currentEnergy += enemyEnergies[i]
                points += 1
                marked_enemies.add(i)

        return points