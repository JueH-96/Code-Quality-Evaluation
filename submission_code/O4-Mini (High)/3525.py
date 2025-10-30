from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        # Base case for hour 0
        dpA = energyDrinkA[0]  # max energy ending by drinking A at hour 0
        dpB = energyDrinkB[0]  # max energy ending by drinking B at hour 0
        dp0 = 0                # max energy ending by cleansing at hour 0
        
        for i in range(1, n):
            a = energyDrinkA[i]
            b = energyDrinkB[i]
            
            # Save previous states
            prevA, prevB, prev0 = dpA, dpB, dp0
            
            # If we drink A at hour i, we must have either been drinking A or been clean at i-1
            dpA = max(prevA, prev0) + a
            # If we drink B at hour i, we must have either been drinking B or been clean at i-1
            dpB = max(prevB, prev0) + b
            # If we cleanse at hour i, we can come from any state at i-1
            dp0 = max(prevA, prevB, prev0)
        
        # The answer is the best of ending in any of the three states
        return max(dpA, dpB, dp0)