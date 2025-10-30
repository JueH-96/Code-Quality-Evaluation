class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # dpA[i] will store the maximum energy boost obtainable up to hour i if we drink from A at hour i
        # dpB[i] will store the maximum energy boost obtainable up to hour i if we drink from B at hour i
        dpA = [0] * n
        dpB = [0] * n
        
        # Initialize the first hour
        dpA[0] = energyDrinkA[0]
        dpB[0] = energyDrinkB[0]
        
        # Fill dp arrays
        for i in range(1, n):
            dpA[i] = max(dpA[i-1] + energyDrinkA[i], dpB[i-1] if i > 1 else 0 + energyDrinkA[i])
            dpB[i] = max(dpB[i-1] + energyDrinkB[i], dpA[i-1] if i > 1 else 0 + energyDrinkB[i])
        
        # The answer is the maximum of the last values in dpA and dpB
        return max(dpA[-1], dpB[-1])