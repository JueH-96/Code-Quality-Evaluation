from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        if n == 0:
            return 0
        if n == 1:
            return max(energyDrinkA[0], energyDrinkB[0])
        
        # Initialize variables for previous two steps for A and B
        prev2_a = 0
        prev1_a = energyDrinkA[0]
        prev2_b = 0
        prev1_b = energyDrinkB[0]
        
        for i in range(1, n):
            # Compute current_a: max of staying on A or switching from B two steps back
            current_a = max(
                prev1_a + energyDrinkA[i],
                (prev2_b if i >= 2 else 0) + energyDrinkA[i]
            )
            
            # Compute current_b: max of staying on B or switching from A two steps back
            current_b = max(
                prev1_b + energyDrinkB[i],
                (prev2_a if i >= 2 else 0) + energyDrinkB[i]
            )
            
            # Update the previous values for next iteration
            # Save the new prev2 and prev1 values
            new_prev2_a, new_prev1_a = prev1_a, current_a
            new_prev2_b, new_prev1_b = prev1_b, current_b
            
            # Update the variables
            prev2_a, prev1_a = new_prev2_a, new_prev1_a
            prev2_b, prev1_b = new_prev2_b, new_prev1_b
        
        return max(prev1_a, prev1_b)