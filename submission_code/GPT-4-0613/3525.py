class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        dp = [[0]*3 for _ in range(n+1)]
        for i in range(n-1, -1, -1):
            dp[i][0] = max(dp[i+1])
            dp[i][1] = energyDrinkA[i] + max(dp[i+1][0], dp[i+1][2])
            dp[i][2] = energyDrinkB[i] + max(dp[i+1][0], dp[i+1][1])
        return max(dp[0])