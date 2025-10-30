class Solution:
    def maxEnergyBoost(self, energyDrinkA: list[int], energyDrinkB: list[int]) -> int:
        n = len(energyDrinkA)
        # dpA[i] will store the max energy boost if we drink A at i-th hour
        # dpB[i] will store the max energy boost if we drink B at i-th hour
        dpA, dpB = [0] * n, [0] * n
        
        # Base cases
        dpA[0] = energyDrinkA[0]
        dpB[0] = energyDrinkB[0]
        
        # If we have only one hour, we can't switch, so we take the max of both
        if n == 1:
            return max(dpA[0], dpB[0])
        
        # For the second hour, we can't switch, so we just add the energy from the same drink
        dpA[1] = dpA[0] + energyDrinkA[1]
        dpB[1] = dpB[0] + energyDrinkB[1]
        
        # If we have only two hours, we can't switch, so we take the max of both
        if n == 2:
            return max(dpA[1], dpB[1])
        
        # Fill dp arrays
        for i in range(2, n):
            # If we drink A at i-th hour, we can either continue with A or switch from B (losing the (i-1)-th hour)
            dpA[i] = max(dpA[i-1] + energyDrinkA[i], dpB[i-2] + energyDrinkA[i])
            # If we drink B at i-th hour, we can either continue with B or switch from A (losing the (i-1)-th hour)
            dpB[i] = max(dpB[i-1] + energyDrinkB[i], dpA[i-2] + energyDrinkB[i])
        
        # The answer will be the max energy boost at the last hour, regardless of the drink
        return max(dpA[n-1], dpB[n-1])

# Example usage:
# sol = Solution()
# print(sol.maxEnergyBoost([1,3,1], [3,1,1]))  # Output: 5
# print(sol.maxEnergyBoost([4,1,1], [1,1,3]))  # Output: 7