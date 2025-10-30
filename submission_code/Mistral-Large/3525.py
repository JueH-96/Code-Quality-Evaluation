from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)

        # Initialize dp arrays
        dpA = [0] * n
        dpB = [0] * n

        # Base case
        dpA[0] = energyDrinkA[0]
        dpB[0] = energyDrinkB[0]

        # Fill dp arrays
        for i in range(1, n):
            # Continue with the same drink
            dpA[i] = max(dpA[i], dpA[i-1] + energyDrinkA[i])
            dpB[i] = max(dpB[i], dpB[i-1] + energyDrinkB[i])

            # Switch to the other drink
            if i > 1:
                dpA[i] = max(dpA[i], dpB[i-2] + energyDrinkA[i])
                dpB[i] = max(dpB[i], dpA[i-2] + energyDrinkB[i])

        # The result is the maximum of the last entries in dpA and dpB
        return max(dpA[-1], dpB[-1])

# Example usage:
# solution = Solution()
# print(solution.maxEnergyBoost([1,3,1], [3,1,1]))  # Output: 5
# print(solution.maxEnergyBoost([4,1,1], [1,1,3]))  # Output: 7