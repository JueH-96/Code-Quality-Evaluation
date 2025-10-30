class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        prev_a = energyDrinkA[0]
        prev_b = energyDrinkB[0]
        prev_none = 0
        
        for i in range(1, n):
            curr_a = max(prev_a, prev_none) + energyDrinkA[i]
            curr_b = max(prev_b, prev_none) + energyDrinkB[i]
            curr_none = max(prev_a, prev_b, prev_none)
            
            prev_a, prev_b, prev_none = curr_a, curr_b, curr_none
        
        return max(prev_a, prev_b, prev_none)