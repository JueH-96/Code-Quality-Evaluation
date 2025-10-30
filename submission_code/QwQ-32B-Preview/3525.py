class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        if n == 0:
            return 0
        
        # Initialize variables for dpA and dpB using the first two hours
        dpA_prev2 = energyDrinkA[0]
        dpB_prev2 = energyDrinkB[0]
        
        if n >= 2:
            dpA_prev = dpA_prev2 + energyDrinkA[1]
            dpB_prev = dpB_prev2 + energyDrinkB[1]
        else:
            # If n == 1, return the maximum of the first hour for A or B
            return max(dpA_prev2, dpB_prev2)
        
        # Iterate from the third hour to the end
        for i in range(2, n):
            dpA_current = max(dpA_prev + energyDrinkA[i], dpB_prev2 + energyDrinkA[i])
            dpB_current = max(dpB_prev + energyDrinkB[i], dpA_prev2 + energyDrinkB[i])
            
            # Update previous values for the next iteration
            dpA_prev2 = dpA_prev
            dpB_prev2 = dpB_prev
            dpA_prev = dpA_current
            dpB_prev = dpB_current
        
        # The answer is the maximum of dpA and dpB for the last hour
        return max(dpA_prev, dpB_prev)