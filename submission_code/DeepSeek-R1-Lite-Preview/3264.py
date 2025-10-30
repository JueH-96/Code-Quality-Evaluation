from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        n = len(enemyEnergies)
        enemyEnergies.sort()
        
        # Compute prefix sums of sorted enemy energies
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + enemyEnergies[i]
        
        total_sum = prefix[n]
        
        # Binary search to find the maximum k such that 2*prefix[k] <= total_sum + currentEnergy
        left, right = 0, n
        result = 0
        while left <= right:
            mid = (left + right) // 2
            if 2 * prefix[mid] <= total_sum + currentEnergy:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
                
        return result