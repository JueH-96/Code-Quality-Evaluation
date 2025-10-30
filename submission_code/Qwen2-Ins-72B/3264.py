class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        n = len(enemyEnergies)
        enemyEnergies.sort()
        dp = [0] * (n + 1)
        dp[0] = 0
        energy = [0] * (n + 1)
        energy[0] = currentEnergy
        for i in range(1, n + 1):
            dp[i] = dp[i - 1]
            energy[i] = energy[i - 1]
            if energy[i] >= enemyEnergies[i - 1]:
                dp[i] = dp[i - 1] + 1
                energy[i] -= enemyEnergies[i - 1]
            if i > 1 and energy[i - 1] >= 1:
                dp[i] = max(dp[i], dp[i - 2] + 1)
                energy[i] = energy[i - 2] + enemyEnergies[i - 1]
        return dp[n]