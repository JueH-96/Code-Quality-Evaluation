class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        points = 0
        for energy in enemyEnergies:
            if currentEnergy >= energy:
                points += 1
                currentEnergy -= energy
            else:
                break
        return points