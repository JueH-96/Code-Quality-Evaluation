class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        if n == 0:
            return 0
        
        prevA = energyDrinkA[0]
        prevB = energyDrinkB[0]
        prevW = 0
        
        for i in range(1, n):
            currA = max(prevA, prevW) + energyDrinkA[i]
            currB = max(prevB, prevW) + energyDrinkB[i]
            currW = max(prevA, prevB, prevW)
            prevA, prevB, prevW = currA, currB, currW
        
        return max(prevA, prevB, prevW)