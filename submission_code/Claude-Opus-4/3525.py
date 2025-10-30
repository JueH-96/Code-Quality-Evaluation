class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # dpA[i] = max energy if drinking A at hour i
        # dpB[i] = max energy if drinking B at hour i
        dpA = [0] * n
        dpB = [0] * n
        
        # Base cases
        dpA[0] = energyDrinkA[0]
        dpB[0] = energyDrinkB[0]
        
        if n > 1:
            dpA[1] = energyDrinkA[0] + energyDrinkA[1]
            dpB[1] = energyDrinkB[0] + energyDrinkB[1]
        
        # Fill the dp arrays
        for i in range(2, n):
            # For dpA[i]: either continue with A or switch from B (lose hour i-1)
            dpA[i] = energyDrinkA[i] + max(dpA[i-1], dpB[i-2])
            
            # For dpB[i]: either continue with B or switch from A (lose hour i-1)
            dpB[i] = energyDrinkB[i] + max(dpB[i-1], dpA[i-2])
        
        # Return the maximum of drinking A or B at the last hour
        return max(dpA[n-1], dpB[n-1])