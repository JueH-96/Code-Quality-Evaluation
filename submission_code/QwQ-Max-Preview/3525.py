from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        if n == 0:
            return 0
        
        prev_a = energyDrinkA[0]
        prev_b = energyDrinkB[0]
        
        for i in range(1, n):
            new_a = max(prev_a + energyDrinkA[i], prev_b)
            new_b = max(prev_b + energyDrinkB[i], prev_a)
            prev_a, prev_b = new_a, new_b
        
        return max(prev_a, prev_b)