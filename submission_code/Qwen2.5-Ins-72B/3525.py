from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        dp = [[0] * 2 for _ in range(n + 1)]
        
        for i in range(n - 1, -1, -1):
            dp[i][0] = max(dp[i + 1][0] + energyDrinkA[i], dp[i + 2][1] + energyDrinkB[i])
            dp[i][1] = max(dp[i + 1][1] + energyDrinkB[i], dp[i + 2][0] + energyDrinkA[i])
        
        return max(dp[0][0], dp[0][1])