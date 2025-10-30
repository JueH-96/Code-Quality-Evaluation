from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # dpA[i]: max total energy gain from hour 0 to i, ending by drinking A at hour i.
        # dpB[i]: max total energy gain from hour 0 to i, ending by drinking B at hour i.
        
        # We use O(1) space by only storing the values for the previous two steps.
        # Let's initialize variables for i=0 and i=1 to bootstrap the process.
        # `..._prev2` will hold dp values for step i-2.
        # `..._prev1` will hold dp values for step i-1.

        # Base case for i=0:
        dpA_prev2 = energyDrinkA[0]
        dpB_prev2 = energyDrinkB[0]
        
        # Base case for i=1:
        # A switch isn't possible yet, so we must continue the drink from hour 0.
        dpA_prev1 = dpA_prev2 + energyDrinkA[1]
        dpB_prev1 = dpB_prev2 + energyDrinkB[1]
        
        # Since the constraint is n >= 3, the loop will run at least once.
        # This implementation would also be correct for n=2, as the loop would
        # not run and the function would correctly return max(dpA_prev1, dpB_prev1).
        
        # Iterate from hour i=2 up to n-1
        for i in range(2, n):
            # Recurrence relation:
            # dpA[i] = energyDrinkA[i] + max(dpA[i-1], dpB[i-2])
            # dpB[i] = energyDrinkB[i] + max(dpB[i-1], dpA[i-2])
            
            # Calculate dpA[i] and dpB[i] for the current hour
            current_dpA = energyDrinkA[i] + max(dpA_prev1, dpB_prev2)
            current_dpB = energyDrinkB[i] + max(dpB_prev1, dpA_prev2)
            
            # Update state variables for the next iteration:
            # The i-1 values become the new i-2 values.
            dpA_prev2 = dpA_prev1
            dpB_prev2 = dpB_prev1
            
            # The current i values become the new i-1 values.
            dpA_prev1 = current_dpA
            dpB_prev1 = current_dpB
            
        # The result is the maximum energy at the last hour (n-1),
        # which is now stored in dpA_prev1 and dpB_prev1.
        return max(dpA_prev1, dpB_prev1)