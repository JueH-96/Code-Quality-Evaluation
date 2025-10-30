import math
# `List` is likely provided by the execution environment, but good practice to import
from typing import List 

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        """
        Calculates the maximum total energy boost achievable in n hours using dynamic programming
        with O(1) space complexity.

        The problem asks for the maximum energy boost over n hours (indexed 0 to n-1).
        We can drink either energy drink A or B each hour. Switching drinks incurs a one-hour
        penalty where no energy is gained.

        We use dynamic programming with four states representing the action taken at the current hour:
        1. Drinking A: Consumed drink A at the current hour.
        2. Drinking B: Consumed drink B at the current hour.
        3. Waiting (B->A): Was consuming B previously, decided to switch to A, so waiting this hour.
        4. Waiting (A->B): Was consuming A previously, decided to switch to B, so waiting this hour.

        Args:
            energyDrinkA: A list of integers representing energy boosts from drink A per hour.
            energyDrinkB: A list of integers representing energy boosts from drink B per hour.

        Returns:
            The maximum total energy boost as an integer.
        """
        n = len(energyDrinkA)

        # Initialize DP state variables. These variables will track the maximum energy accumulated
        # up to the *previous* hour, ending in each of the four possible states.
        # We initialize these based on the state after the first hour (hour 0).
        
        # Base case: state after hour 0
        # At hour 0, we can start with either drink A or B without penalty.
        prev_dp_A = energyDrinkA[0]  # Max energy after hour 0, ending with drinking A
        prev_dp_B = energyDrinkB[0]  # Max energy after hour 0, ending with drinking B
        
        # Waiting states are impossible at hour 0 as we just started.
        # Initialize with negative infinity to ensure they are not chosen unless necessary later.
        prev_dp_WaitA = -math.inf    # Max energy after hour 0, ending with waiting (B->A switch)
        prev_dp_WaitB = -math.inf    # Max energy after hour 0, ending with waiting (A->B switch)
        
        # Iterate from the second hour (index 1) up to the last hour (index n-1)
        for i in range(1, n):
            # Calculate DP states for the end of the current hour (i) using values from hour (i-1).
            # We use temporary variables (`current_dp_...`) to store the results for hour i.
            # This prevents overwriting values from hour i-1 that are still needed for other calculations
            # within this iteration for hour i.
            
            # Max energy ending with drinking A at hour i:
            # To drink A at hour i, we could have either drunk A at hour i-1,
            # or finished waiting at hour i-1 after switching from B to A.
            # We take the max energy from these possibilities and add the energy from drink A at hour i.
            current_dp_A = max(prev_dp_A, prev_dp_WaitA) + energyDrinkA[i]
            
            # Max energy ending with drinking B at hour i:
            # Similar logic as for drinking A. We could have come from state B or state WaitB at hour i-1.
            current_dp_B = max(prev_dp_B, prev_dp_WaitB) + energyDrinkB[i]
            
            # Max energy ending with waiting at hour i (due to deciding to switch B->A):
            # To be waiting at hour i because of a B->A switch, we must have been drinking B at hour i-1.
            # We decided to switch, so we wait this hour (hour i) and gain 0 energy.
            # The total accumulated energy is the same as the energy accumulated ending with state B at hour i-1.
            current_dp_WaitA = prev_dp_B
            
            # Max energy ending with waiting at hour i (due to deciding to switch A->B):
            # Similar logic as for WaitA. We must have been drinking A at hour i-1.
            # The total accumulated energy is the same as ending with state A at hour i-1.
            current_dp_WaitB = prev_dp_A
            
            # Update the 'previous' state variables to the computed 'current' state values.
            # These updated values will be used as the 'previous' state for the next iteration (hour i+1).
            prev_dp_A = current_dp_A
            prev_dp_B = current_dp_B
            prev_dp_WaitA = current_dp_WaitA
            prev_dp_WaitB = current_dp_WaitB
            
        # After the loop finishes, the 'prev_' variables hold the maximum energy accumulated
        # up to the end of the last hour (n-1), for each possible final state.
        # The overall maximum energy boost is the maximum value among these four final states.
        final_max_energy = max(prev_dp_A, prev_dp_B, prev_dp_WaitA, prev_dp_WaitB)
        
        # Return the final result. Since energy boosts are positive integers, the result
        # will be a non-negative integer. Python's integers handle large values automatically.
        # Casting to int ensures the return type matches the specified `-> int`.
        return int(final_max_energy)