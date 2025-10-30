class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        # We'll track three states at each hour:
        # prevA   = max total energy if we ended drinking A in the previous hour
        # prevB   = max total energy if we ended drinking B in the previous hour
        # prevSkip = max total energy if we skipped the previous hour
        prevA = 0
        prevB = 0
        prevSkip = 0
        
        for i in range(n):
            curSkip = max(prevA, prevB, prevSkip)
            curA = max(prevA, prevSkip) + energyDrinkA[i]
            curB = max(prevB, prevSkip) + energyDrinkB[i]
            prevA, prevB, prevSkip = curA, curB, curSkip
        
        # The answer is the maximum of ending with A, ending with B, or skipping the last hour
        return max(prevA, prevB, prevSkip)