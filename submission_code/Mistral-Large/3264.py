from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        # Sort the enemy energies
        enemyEnergies.sort()

        points = 0
        total_marked_energy = 0

        # Use a two-pointer approach to decide which enemies to mark
        left = 0
        right = len(enemyEnergies) - 1

        while left <= right:
            if currentEnergy >= enemyEnergies[left]:
                # Fight the enemy with the least energy
                currentEnergy -= enemyEnergies[left]
                points += 1
                left += 1
            elif points > 0:
                # Mark the enemy with the highest energy
                total_marked_energy += enemyEnergies[right]
                points -= 1
                right -= 1
                currentEnergy += total_marked_energy
            else:
                break

        return points