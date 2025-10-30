class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        n = len(enemyEnergies)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + enemyEnergies[i]
        total_sum = prefix[n]
        max_points = 0
        for k in range(n + 1):
            sumA = prefix[k]
            sumB = total_sum - sumA
            if sumA <= currentEnergy + sumB:
                max_points = max(max_points, k)
        return max_points