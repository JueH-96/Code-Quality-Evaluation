class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # dpA[i] = max total energy if we use drink A at hour i
        # dpB[i] = max total energy if we use drink B at hour i
        dpA = [0] * n
        dpB = [0] * n
        
        # Initialize the first hour
        dpA[0] = energyDrinkA[0]
        dpB[0] = energyDrinkB[0]
        
        for i in range(1, n):
            # If we continue with A, add energy from A at hour i;
            # or if we switch from B to A, we gain no energy at i (hence dpB[i-1]).
            dpA[i] = max(dpA[i-1] + energyDrinkA[i], dpB[i-1])
            
            # Similarly for B:
            dpB[i] = max(dpB[i-1] + energyDrinkB[i], dpA[i-1])
        
        return max(dpA[n-1], dpB[n-1])