from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        # This is the "bag of tokens" pattern:
        # Sort the energies, use two‐pointer greedy:
        # - If we can "fight" (currentEnergy >= smallest), do so to gain one point.
        # - Otherwise, if we have at least one point, we "rest" (sacrifice one point)
        #   to gain the largest remaining energy.
        # Track the max points seen.
        enemyEnergies.sort()
        l, r = 0, len(enemyEnergies) - 1
        points = 0
        max_points = 0
        
        while l <= r:
            # fight the cheapest enemy if we can
            if currentEnergy >= enemyEnergies[l]:
                currentEnergy -= enemyEnergies[l]
                points += 1
                l += 1
                max_points = max(max_points, points)
            # otherwise, if we have a point to spend, rest to get back big energy
            elif points > 0:
                # give up one point to gain energy of the most expensive still unmarked enemy
                currentEnergy += enemyEnergies[r]
                points -= 1
                r -= 1
            else:
                # can't fight (not enough energy) and can't rest (no points)
                break
        
        return max_points