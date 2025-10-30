class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # Initialize dp arrays
        dp_a = [0] * n  # Max energy boost ending with drink A
        dp_b = [0] * n  # Max energy boost ending with drink B
        
        # Base cases
        dp_a[0] = energyDrinkA[0]
        dp_b[0] = energyDrinkB[0]
        
        if n > 1:
            dp_a[1] = energyDrinkA[0] + energyDrinkA[1]
            dp_b[1] = energyDrinkB[0] + energyDrinkB[1]
        
        # Fill dp arrays
        for i in range(2, n):
            dp_a[i] = max(dp_a[i-1] + energyDrinkA[i], dp_b[i-2] + energyDrinkA[i])
            dp_b[i] = max(dp_b[i-1] + energyDrinkB[i], dp_a[i-2] + energyDrinkB[i])
        
        # Return the maximum of the last elements in both dp arrays
        return max(dp_a[-1], dp_b[-1])