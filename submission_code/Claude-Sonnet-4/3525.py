class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        if n == 1:
            return max(energyDrinkA[0], energyDrinkB[0])
        
        # dpA[i] represents max energy ending at hour i drinking A
        # dpB[i] represents max energy ending at hour i drinking B
        dpA = [0] * n
        dpB = [0] * n
        
        # Base cases
        dpA[0] = energyDrinkA[0]
        dpB[0] = energyDrinkB[0]
        
        dpA[1] = energyDrinkA[1] + energyDrinkA[0]
        dpB[1] = energyDrinkB[1] + energyDrinkB[0]
        
        # Fill the dp arrays
        for i in range(2, n):
            # To drink A at hour i, either continue from A or switch from B (skip previous hour)
            dpA[i] = energyDrinkA[i] + max(dpA[i-1], dpB[i-2])
            # To drink B at hour i, either continue from B or switch from A (skip previous hour)  
            dpB[i] = energyDrinkB[i] + max(dpB[i-1], dpA[i-2])
        
        return max(dpA[n-1], dpB[n-1])