class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)

        # Initialize dp arrays
        dpA = [0] * n
        dpB = [0] * n

        # Base cases
        dpA[0] = energyDrinkA[0]
        dpB[0] = energyDrinkB[0]

        # Fill dp arrays
        for i in range(1, n):
            dpA[i] = max(dpA[i-1] + energyDrinkA[i], dpB[i-1] + energyDrinkA[i])
            dpB[i] = max(dpB[i-1] + energyDrinkB[i], dpA[i-1] + energyDrinkB[i])

        # The answer is the maximum value in the last cell of either dpA or dpB
        return max(dpA[-1], dpB[-1])