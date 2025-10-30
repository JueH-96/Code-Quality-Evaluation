class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        n = len(enemyEnergies)
        enemyEnergies.sort()

        max_points = 0
        for p in range(n + 1):
            if p == 0:
                max_points = max(max_points, 0)
                continue

            if p <= n:
                required_energy = sum(enemyEnergies[:p])
                if currentEnergy >= required_energy:
                    max_points = max(max_points, p)

        return max_points