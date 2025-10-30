from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        n = len(enemyEnergies)
        suffix_sum = [0] * (n + 1)
        for i in range(n-1, -1, -1):
            suffix_sum[i] = suffix_sum[i+1] + enemyEnergies[i]
        
        max_points = 0
        for k in range(0, n+1):
            sum_marked = suffix_sum[n - k]  # sum of the largest k enemies
            total_energy = currentEnergy + sum_marked
            m = n - k
            sum_points = 0
            # Compute sum of total_energy // E_i for the first m enemies
            for i in range(m):
                sum_points += total_energy // enemyEnergies[i]
            if sum_points >= k:
                max_points = max(max_points, sum_points - k)
        return max_points