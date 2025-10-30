from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # dp[i][0] means the max energy boost up to hour i if we drink A at hour i
        # dp[i][1] means the max energy boost up to hour i if we drink B at hour i
        dp = [[0, 0] for _ in range(n)]
        
        # Initialize the first hour
        dp[0][0] = energyDrinkA[0]
        dp[0][1] = energyDrinkB[0]
        
        for i in range(1, n):
            # If we drink A at hour i, we could have come from drinking A or B at hour i-2
            dp[i][0] = energyDrinkA[i] + max(dp[i-1][0], dp[i-1][1] if i > 1 else 0)
            # If we drink B at hour i, we could have come from drinking A or B at hour i-2
            dp[i][1] = energyDrinkB[i] + max(dp[i-1][1], dp[i-1][0] if i > 1 else 0)
        
        # The result is the maximum energy boost we can get by the end of the last hour
        return max(dp[n-1][0], dp[n-1][1])