from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        if n == 0:
            return 0
        a = energyDrinkA[0]
        b = energyDrinkB[0]
        for i in range(1, n):
            new_a = max(a + energyDrinkA[i], b)
            new_b = max(b + energyDrinkB[i], a)
            a, b = new_a, new_b
        return max(a, b)