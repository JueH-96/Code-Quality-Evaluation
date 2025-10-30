class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # dpA[i] will store the maximum energy boost we can get up to hour i if we end with drink A
        # dpB[i] will store the maximum energy boost we can get up to hour i if we end with drink B
        dpA = [0] * n
        dpB = [0] * n
        
        # Initialize the first hour
        dpA[0] = energyDrinkA[0]
        dpB[0] = energyDrinkB[0]
        
        for i in range(1, n):
            # If we choose drink A at hour i, we can either continue from drink A at hour i-1
            # or switch from drink B at hour i-1 (which means we skip hour i-1)
            dpA[i] = max(dpA[i-1] + energyDrinkA[i], dpB[i-1])
            
            # If we choose drink B at hour i, we can either continue from drink B at hour i-1
            # or switch from drink A at hour i-1 (which means we skip hour i-1)
            dpB[i] = max(dpB[i-1] + energyDrinkB[i], dpA[i-1])
        
        # The result is the maximum energy boost we can get at the last hour
        return max(dpA[-1], dpB[-1])