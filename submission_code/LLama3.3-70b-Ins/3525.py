from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # Initialize dp arrays to store the maximum energy boost at each hour
        # dpA[i] represents the maximum energy boost if we drink energy drink A at hour i
        # dpB[i] represents the maximum energy boost if we drink energy drink B at hour i
        dpA = [0] * n
        dpB = [0] * n
        
        # Base cases
        dpA[0] = energyDrinkA[0]
        dpB[0] = energyDrinkB[0]
        
        # Fill up dp arrays
        for i in range(1, n):
            # If we drink energy drink A at hour i, we can either drink A at hour i-1 or B at hour i-2
            dpA[i] = max(dpA[i-1], dpB[i-1] + energyDrinkA[i] if i > 1 else energyDrinkA[i])
            # If we drink energy drink B at hour i, we can either drink B at hour i-1 or A at hour i-2
            dpB[i] = max(dpB[i-1], dpA[i-1] + energyDrinkB[i] if i > 1 else energyDrinkB[i])
        
        # The maximum energy boost is the maximum of dpA[n-1] and dpB[n-1]
        return max(dpA[-1], dpB[-1])