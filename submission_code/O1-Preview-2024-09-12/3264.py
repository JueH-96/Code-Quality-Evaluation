class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        from collections import deque
        enemyDeque = deque(enemyEnergies)
        points = 0
        maxPoints = 0
        while enemyDeque:
            # Perform first operation as much as possible
            while enemyDeque and currentEnergy >= enemyDeque[0]:
                currentEnergy -= enemyDeque[0]
                points += 1
                maxPoints = max(maxPoints, points)
                # First operation does not remove the enemy as it's not marked
            if points >=1 and enemyDeque:
                # Perform second operation to gain energy
                currentEnergy += enemyDeque.pop()
                # Enemy is now marked
            else:
                break
        return maxPoints