from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        points = 0
        while True:
            # Try to defeat an enemy to gain a point
            defeated = False
            for i, energy in enumerate(enemyEnergies):
                if currentEnergy >= energy:
                    currentEnergy -= energy
                    points += 1
                    defeated = True
                    break
            if not defeated:
                break

            # If we have at least one point, try to gain energy by marking an enemy
            if points > 0:
                max_energy_gain = 0
                for energy in enemyEnergies:
                    if energy > max_energy_gain:
                        max_energy_gain = energy
                currentEnergy += max_energy_gain

        return points