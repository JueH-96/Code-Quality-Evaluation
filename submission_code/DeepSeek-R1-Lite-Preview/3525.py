class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # Initialize the first hour
        prev_A = energyDrinkA[0]
        prev_B = energyDrinkB[0]
        prev_cleanse = 0  # Not applicable at the start, set to 0
        
        for i in range(1, n):
            current_A = max(prev_A + energyDrinkA[i], prev_cleanse + energyDrinkA[i])
            current_B = max(prev_B + energyDrinkB[i], prev_cleanse + energyDrinkB[i])
            current_cleanse = max(prev_A, prev_B)
            
            prev_A, prev_B, prev_cleanse = current_A, current_B, current_cleanse
        
        return max(prev_A, prev_B, prev_cleanse)