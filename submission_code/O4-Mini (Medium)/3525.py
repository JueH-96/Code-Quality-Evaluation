from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        # dpA = max energy if we drink A at the current hour
        # dpB = max energy if we drink B at the current hour
        # dpC = max energy if we are cleansing (i.e., we switched) at the current hour (gain 0 energy)
        dpA = energyDrinkA[0]
        dpB = energyDrinkB[0]
        dpC = float('-inf')  # cannot be cleansing at hour 0
        
        for i in range(1, n):
            a = energyDrinkA[i]
            b = energyDrinkB[i]
            
            # If we drink A now, we must have been either drinking A or cleansing last hour
            newA = max(dpA, dpC) + a
            # If we drink B now, we must have been either drinking B or cleansing last hour
            newB = max(dpB, dpC) + b
            # If we are cleansing now, we must have drunk the opposite drink last hour
            newC = max(dpA, dpB)
            
            dpA, dpB, dpC = newA, newB, newC
        
        # The answer is the best we can be in at the last hour (drinking A, drinking B, or cleansing)
        return max(dpA, dpB, dpC)