class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        # Initialize DP arrays
        # dpA[i]: max energy if we end with A at hour i
        # dpB[i]: max energy if we end with B at hour i
        dpA = [0] * n
        dpB = [0] * n
        dpA[0] = energyDrinkA[0]
        dpB[0] = energyDrinkB[0]
        
        for i in range(1, n):
            # For dpA[i], we can either continue with A or switch from B (with a gap)
            dpA[i] = max(dpA[i-1] + energyDrinkA[i], dpB[i-1] + energyDrinkA[i])
            # For dpB[i], we can either continue with B or switch from A (with a gap)
            dpB[i] = max(dpB[i-1] + energyDrinkB[i], dpA[i-1] + energyDrinkB[i])
        
        # The maximum energy is the maximum of the last elements in dpA and dpB
        return max(dpA[-1], dpB[-1])