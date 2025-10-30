class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        dp = [[0, 0] for _ in range(n + 1)]
        dp[0] = [0, 0]

        for i in range(n):
            dp[i + 1][0] = max(dp[i][0], dp[i][1]) + energyDrinkA[i]
            dp[i + 1][1] = max(dp[i][0], dp[i][1]) + energyDrinkB[i]

        return max(dp[n][0], dp[n][1])