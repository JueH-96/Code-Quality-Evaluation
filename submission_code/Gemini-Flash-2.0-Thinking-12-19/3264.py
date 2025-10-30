from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        n = len(enemyEnergies)
        memo = {}

        def solve(current_energy, points, marked_mask_tuple):
            if (current_energy, points, marked_mask_tuple) in memo:
                return memo[(current_energy, points, marked_mask_tuple)]

            max_points_here = points
            marked = list(marked_mask_tuple)
            for i in range(n):
                if not marked[i]:
                    # Try to attack
                    if current_energy >= enemyEnergies[i]:
                        marked[i] = True
                        max_points_here = max(max_points_here, solve(current_energy - enemyEnergies[i], points + 1, tuple(marked)))
                        marked[i] = False # Backtrack

                    # Try to recover if points >= 1
                    if points >= 1:
                        marked[i] = True
                        max_points_here = max(max_points_here, solve(current_energy + enemyEnergies[i], points, tuple(marked)))
                        marked[i] = False # Backtrack

            memo[(current_energy, points, marked_mask_tuple)] = max_points_here
            return max_points_here

        marked_init = [False] * n
        result = solve(currentEnergy, 0, tuple(marked_init))
        return result