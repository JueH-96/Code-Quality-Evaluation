class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # dpA[i]:   max total if we consume drink A at hour i
        # dpB[i]:   max total if we consume drink B at hour i
        # dpSkip[i]: max total if we skip hour i
        dpA = [0]*n
        dpB = [0]*n
        dpSkip = [0]*n
        
        # Initialize base case for hour 0
        dpA[0] = energyDrinkA[0]
        dpB[0] = energyDrinkB[0]
        dpSkip[0] = 0
        
        for i in range(1, n):
            dpA[i] = max(dpA[i-1], dpSkip[i-1]) + energyDrinkA[i]
            dpB[i] = max(dpB[i-1], dpSkip[i-1]) + energyDrinkB[i]
            dpSkip[i] = max(dpA[i-1], dpB[i-1], dpSkip[i-1])
        
        return max(dpA[-1], dpB[-1], dpSkip[-1])