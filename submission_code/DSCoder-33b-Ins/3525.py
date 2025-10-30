class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        dp = [[0, 0] for _ in range(n+1)]
        dp[0][1] = float('-inf')
        for i in range(1, n+1):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + energyDrinkA[i-1])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + energyDrinkB[i-1])
        return max(dp[n])