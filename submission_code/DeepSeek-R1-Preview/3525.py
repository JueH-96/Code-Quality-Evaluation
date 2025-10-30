from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        if n == 0:
            return 0
        prev_a = energyDrinkA[0]
        prev_b = energyDrinkB[0]
        prev_c = 0
        for i in range(1, n):
            current_a = max(prev_a, prev_c) + energyDrinkA[i]
            current_b = max(prev_b, prev_c) + energyDrinkB[i]
            current_c = max(prev_a, prev_b)
            prev_a, prev_b, prev_c = current_a, current_b, current_c
        return max(prev_a, prev_b, prev_c)