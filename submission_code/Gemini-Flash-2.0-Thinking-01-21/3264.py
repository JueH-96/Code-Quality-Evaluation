from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        n = len(enemyEnergies)
        memo = {}
        
        def solve(current_energy, points, marked_mask):
            state = (current_energy, points, marked_mask)
            if state in memo:
                return memo[state]
            
            max_points_obtained = points
            
            for i in range(n):
                if (marked_mask & (1 << i)) == 0: # if enemy i is not marked
                    # Attack operation
                    if current_energy >= enemyEnergies[i]:
                        next_energy_attack = current_energy - enemyEnergies[i]
                        points_after_attack = solve(next_energy_attack, points + 1, marked_mask)
                        max_points_obtained = max(max_points_obtained, points_after_attack)
                    # Recover operation
                    if points >= 1:
                        next_energy_recover = current_energy + enemyEnergies[i]
                        next_marked_mask = marked_mask | (1 << i)
                        points_after_recover = solve(next_energy_recover, points, next_marked_mask)
                        max_points_obtained = max(max_points_obtained, points_after_recover)
                        
            memo[state] = max_points_obtained
            return max_points_obtained
            
        return solve(currentEnergy, 0, 0)