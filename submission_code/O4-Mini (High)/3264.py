from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        """
        Since marking an enemy never costs points (only requires you have at least one
        point), the optimal strategy is:
          1. Fight the cheapest enemy once (if currentEnergy < cheapest, you can't get any points).
          2. Mark all other enemies to bulk up your energy.
          3. Fight the cheapest enemy as many more times as your total energy allows.
        You can fight the same unmarked enemy repeatedly because fighting does not mark it.
        """
        if not enemyEnergies:
            return 0
        
        # find the smallest energy enemy
        m = min(enemyEnergies)
        # if we can't even fight it once, we get 0 points
        if currentEnergy < m:
            return 0
        
        # total energy in all enemies
        total = sum(enemyEnergies)
        # after one fight of the smallest, we have currentEnergy - m,
        # then we mark all others to add (total - m),
        # so final energy pool = currentEnergy + total - m.
        # Each fight of the smallest costs m energy and yields 1 point.
        # Total points = floor((currentEnergy + total - m) / m)
        return (currentEnergy + total - m) // m