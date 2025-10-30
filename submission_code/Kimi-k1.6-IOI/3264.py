class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        n = len(enemyEnergies)
        if n == 0:
            return 0
        
        # Sort the enemy energies in ascending order
        enemyEnergies.sort()
        
        # Compute prefix sum array
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + enemyEnergies[i]
        total_sum = prefix[-1]
        
        max_points = 0
        
        for k in range(n + 1):
            if k == 0:
                candidate = 0
            else:
                e_k = enemyEnergies[k - 1]
                sum_marked = total_sum - prefix[k]
                if currentEnergy >= enemyEnergies[0]:
                    available_energy = currentEnergy - enemyEnergies[0] + sum_marked
                    attacks = available_energy // e_k
                    candidate = 1 + attacks
                else:
                    candidate = 0
            if candidate > max_points:
                max_points = candidate
        
        return max_points