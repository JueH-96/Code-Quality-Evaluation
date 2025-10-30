from typing import List
import math

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)

        # O(1) space DP state variables
        # These represent the maximum energy up to the *previous* hour (i-1)
        # prev_max_a: max energy ending by consuming A at hour i-1
        # prev_max_b: max energy ending by consuming B at hour i-1
        # prev_max_c: max energy ending with a cleansing hour at hour i-1

        # Base case: Hour 0
        # We can start with either A or B. A cleansing hour is not possible at the start.
        prev_max_a = energyDrinkA[0]
        prev_max_b = energyDrinkB[0]
        prev_max_c = -math.inf # Use -inf to represent an unreachable state

        # Iterate from the second hour (index 1) up to the last hour (index n-1)
        for i in range(1, n):
            # Calculate the maximum energy up to the *current* hour (i)
            
            # To end by consuming A at hour i (curr_max_a):
            # We must have been in a state at hour i-1 from which we can consume A at hour i.
            # This can be:
            # 1. Consumed A at hour i-1 (prev_max_a), continue A at hour i.
            # 2. Had a cleansing hour at hour i-1 (prev_max_c), start A at hour i.
            # Note: Cannot switch directly from B at i-1 to A at i. That requires a cleansing hour at i.
            curr_max_a = max(prev_max_a, prev_max_c) + energyDrinkA[i]

            # To end by consuming B at hour i (curr_max_b):
            # Similar logic as above, but for drink B.
            # 1. Consumed B at hour i-1 (prev_max_b), continue B at hour i.
            # 2. Had a cleansing hour at hour i-1 (prev_max_c), start B at hour i.
            # Note: Cannot switch directly from A at i-1 to B at i. That requires a cleansing hour at i.
            curr_max_b = max(prev_max_b, prev_max_c) + energyDrinkB[i]

            # To have a cleansing hour at hour i (curr_max_c):
            # We must have consumed a drink at hour i-1 and decided to switch at hour i.
            # This transition costs 0 energy at hour i.
            # 1. Consumed A at hour i-1 (prev_max_a) and decided to switch.
            # 2. Consumed B at hour i-1 (prev_max_b) and decided to switch.
            # The max energy up to hour i ending in a cleanse is the max energy up to hour i-1 ending in A or B.
            curr_max_c = max(prev_max_a, prev_max_b)

            # Update the previous hour's max energies for the next iteration (i+1)
            prev_max_a = curr_max_a
            prev_max_b = curr_max_b
            prev_max_c = curr_max_c

        # After iterating through all hours up to n-1, the maximum total energy boost
        # is the maximum energy achieved ending in any of the three states at the final hour (n-1).
        # It's possible the optimal strategy ends with a cleansing hour, e.g., if the last drink options are very low.
        return max(prev_max_a, prev_max_b, prev_max_c)