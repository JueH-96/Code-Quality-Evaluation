from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        # dpA and dpB hold the maximum energy boost so far ending with a consumption hour
        # from drink A and drink B respectively.
        # When you switch, you lose the current hour, so we use "skip" states to capture that decision.
        # skipA: you switched in the previous hour from drink B to A (so current cooldown intended for A)
        # skipB: you switched from drink A to B (cooldown intended for B)
        dpA = energyDrinkA[0]
        dpB = energyDrinkB[0]
        # Initially, we haven't made any switch so no cooldown state exists.
        neg_inf = float("-inf")
        skipA = neg_inf
        skipB = neg_inf
        
        for i in range(1, n):
            # If you choose to drink from A this hour, you could:
            #   (a) have been on A last hour; or
            #   (b) have just finished a cooldown from switching to A.
            new_dpA = max(dpA, skipA) + energyDrinkA[i]
            # Similarly for drinking from B.
            new_dpB = max(dpB, skipB) + energyDrinkB[i]
            
            # If you decide to switch from drink A to drink B,
            # you forfeit the current hour; we store this in skipB (target is B next).
            new_skipB = dpA
            # Likewise, switching from drink B to drink A.
            new_skipA = dpB
            
            # Update states for the current hour i.
            dpA, dpB = new_dpA, new_dpB
            skipA, skipB = new_skipA, new_skipB
            
        # The answer is the best boost that ends with a consumption hour (either drink)
        return max(dpA, dpB)