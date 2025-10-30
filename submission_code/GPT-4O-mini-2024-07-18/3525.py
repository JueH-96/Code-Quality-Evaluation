from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # dp[i][0] = max energy boost up to hour i if we drink A at hour i
        # dp[i][1] = max energy boost up to hour i if we drink B at hour i
        dp = [[0, 0] for _ in range(n)]
        
        # Initialize the first hour
        dp[0][0] = energyDrinkA[0]
        dp[0][1] = energyDrinkB[0]
        
        for i in range(1, n):
            # If we drink A at hour i
            dp[i][0] = max(dp[i-1][0] + energyDrinkA[i], dp[i-1][1] + energyDrinkA[i])
            # If we drink B at hour i
            dp[i][1] = max(dp[i-1][1] + energyDrinkB[i], dp[i-1][0] + energyDrinkB[i])
            
            # If we switch drinks, we need to account for the cleanse hour
            if i > 1:
                dp[i][0] = max(dp[i][0], dp[i-2][0] + energyDrinkA[i])
                dp[i][1] = max(dp[i][1], dp[i-2][1] + energyDrinkB[i])
        
        # The result is the maximum energy boost we can have at the last hour
        return max(dp[n-1][0], dp[n-1][1])