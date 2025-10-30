class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        # Sort the enemy energies in ascending order
        enemyEnergies.sort()
        
        points = 0
        left = 0
        right = len(enemyEnergies) - 1
        
        # We will greedily use the smallest-cost enemies (left side) for "operation 1"
        # and when we can no longer afford to do operation 1, we use the largest-cost
        # enemy (right side) for "operation 2" (if we have at least 1 point).
        while left <= right:
            # Try operation 1 on the smallest unmarked enemy
            if currentEnergy >= enemyEnergies[left]:
                # We can repeatedly do operation 1 on this same enemy
                # as long as we have enough energy.
                cost = enemyEnergies[left]
                times = currentEnergy // cost
                points += times
                currentEnergy -= times * cost
            else:
                # If we can't do operation 1, and we have at least 1 point,
                # do operation 2 (gain energy from the largest cost enemy).
                if points > 0:
                    currentEnergy += enemyEnergies[right]
                    right -= 1
                else:
                    # We can't do operation 1, and have no points to do operation 2
                    break
        
        return points