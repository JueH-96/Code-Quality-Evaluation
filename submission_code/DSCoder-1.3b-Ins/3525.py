class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        dp = [[0]*n for _ in range(n)]
        dp2 = [[0]*n for _ in range(n)]
        dp[0][0] = energyDrinkA[0]
        dp2[0][0] = energyDrinkB[0]
        for i in range(1, n):
            dp[i][0] = max(dp[i-1][0], energyDrinkA[i])
            dp[i][i] = max(dp[i-1][i-1], energyDrinkB[i])
            dp2[i][i-1] = max(dp2[i-1][i-2], energyDrinkB[i])
            dp2[i][0] = max(dp2[i-1][0], energyDrinkA[i])
            for j in range(1, i):
                dp[i][j] = max(dp[i-1][j-1], dp[i-1][j], energyDrinkA[i])
                dp2[i][j] = max(dp2[i-1][j-1], dp2[i-1][j], energyDrinkB[i])
        return max(max(dp[-1]), dp2[-1])