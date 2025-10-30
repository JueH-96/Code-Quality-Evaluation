class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        current = currentEnergy
        points = 0
        i = 0
        n = len(enemyEnergies)
        suffix_sum = [0] * (n + 1)  # suffix_sum[i] = sum(enemyEnergies[i:]) for i in 0..n-1, and 0 at i=n
        
        # Precompute suffix sums
        for i in range(n-1, -1, -1):
            suffix_sum[i] = suffix_sum[i+1] + enemyEnergies[i]
        
        while i < n:
            # Take as much via op1 as possible
            while i < n and current >= enemyEnergies[i]:
                points += 1
                current -= enemyEnergies[i]
                i += 1
            if i >= n:
                break
            
            # Take as much via op2 as possible
            if points > 0:
                sum_op2 = suffix_sum[i]
                current += sum_op2
                points += (n - i)
                i = n
            else:
                break
        
        return points