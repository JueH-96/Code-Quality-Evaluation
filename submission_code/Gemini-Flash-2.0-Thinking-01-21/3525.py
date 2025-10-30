class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        if n == 0:
            return 0
        dp = [[0] * 2 for _ in range(n)]
        dp[0][0] = energyDrinkA[0]
        dp[0][1] = energyDrinkB[0]
        if n >= 2:
            dp[1][0] = energyDrinkA[1] + dp[0][0]
            dp[1][1] = energyDrinkB[1] + dp[0][1]
        for i in range(2, n):
            dp[i][0] = energyDrinkA[i] + max(dp[i-1][0], dp[i-2][1])
            dp[i][1] = energyDrinkB[i] + max(dp[i-1][1], dp[i-2][0])
        return max(dp[n-1][0], dp[n-1][1])