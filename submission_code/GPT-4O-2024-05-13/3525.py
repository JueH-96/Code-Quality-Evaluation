class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # dpA[i] will store the maximum energy boost if we end at hour i with drink A
        # dpB[i] will store the maximum energy boost if we end at hour i with drink B
        dpA = [0] * n
        dpB = [0] * n
        
        # Initialize the first hour
        dpA[0] = energyDrinkA[0]
        dpB[0] = energyDrinkB[0]
        
        for i in range(1, n):
            dpA[i] = max(dpA[i-1] + energyDrinkA[i], dpB[i-1])
            dpB[i] = max(dpB[i-1] + energyDrinkB[i], dpA[i-1])
        
        return max(dpA[-1], dpB[-1])