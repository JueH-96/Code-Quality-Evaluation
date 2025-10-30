class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        n = len(enemyEnergies)
        suffix_sums = [0] * (n + 1)
        for k in range(1, n + 1):
            suffix_sums[k] = suffix_sums[k - 1] + enemyEnergies[-k]
        max_points = 0
        for k in range(0, n + 1):
            sum_marked = suffix_sums[k]
            new_energy = currentEnergy + sum_marked
            m = n - k
            if m <= 0:
                points = 0
            else:
                points = 0
                for i in range(m):
                    points += new_energy // enemyEnergies[i]
            if points > max_points:
                max_points = points
        return max_points