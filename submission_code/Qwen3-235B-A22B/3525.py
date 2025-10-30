from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        if n == 0:
            return 0
        
        prev_a = energyDrinkA[0]
        prev_b = energyDrinkB[0]
        prev_wait = float('-inf')
        
        for i in range(1, n):
            current_a = max(prev_a + energyDrinkA[i], prev_wait + energyDrinkA[i])
            current_b = max(prev_b + energyDrinkB[i], prev_wait + energyDrinkB[i])
            current_wait = max(prev_a, prev_b)
            
            prev_a, prev_b, prev_wait = current_a, current_b, current_wait
        
        return max(prev_a, prev_b, prev_wait)