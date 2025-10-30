class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        if not energyDrinkA:
            return 0
        prevA = energyDrinkA[0]
        prevB = energyDrinkB[0]
        for i in range(1, len(energyDrinkA)):
            currentA = max(prevA, prevB + energyDrinkA[i])
            currentB = max(prevB, prevA + energyDrinkB[i])
            prevA, prevB = currentA, currentB
        return max(prevA, prevB)