from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        a = energyDrinkA[0]
        b = energyDrinkB[0]
        p = 0  # Penalty state starts with 0 since first hour can't have a penalty
        
        for i in range(1, n):
            new_a = max(a + energyDrinkA[i], p + energyDrinkA[i])
            new_b = max(b + energyDrinkB[i], p + energyDrinkB[i])
            new_p = max(a, b)
            a, b, p = new_a, new_b, new_p
        
        return max(a, b, p)