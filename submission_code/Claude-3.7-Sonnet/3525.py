class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # State definitions:
        # drinkA[i] = Maximum boost we can get up to the ith hour if we drink A for the ith hour.
        # drinkB[i] = Maximum boost we can get up to the ith hour if we drink B for the ith hour.
        drinkA = [0] * n
        drinkB = [0] * n
        
        # Base cases
        drinkA[0] = energyDrinkA[0]
        drinkB[0] = energyDrinkB[0]
        
        # For the first transition, since we can't switch yet
        drinkA[1] = drinkA[0] + energyDrinkA[1]
        drinkB[1] = drinkB[0] + energyDrinkB[1]
        
        # DP transitions
        for i in range(2, n):
            # Either stick with A or switch from B (skipping the previous hour)
            drinkA[i] = max(drinkA[i-1] + energyDrinkA[i], drinkB[i-2] + energyDrinkA[i])
            # Either stick with B or switch from A (skipping the previous hour)
            drinkB[i] = max(drinkB[i-1] + energyDrinkB[i], drinkA[i-2] + energyDrinkB[i])
        
        # Return the maximum of the two possibilities for the last hour
        return max(drinkA[n-1], drinkB[n-1])