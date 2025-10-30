import math
from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)

        # dpA: Maximum energy boost up to the current hour, ending by drinking A.
        # dpB: Maximum energy boost up to the current hour, ending by drinking B.
        # dpC: Maximum energy boost up to the current hour, ending by cleansing (0 boost this hour).

        # Initialize base cases for hour 0
        # At hour 0, we must start with a drink. Cleansing is not a valid way to gain energy initially.
        prev_dpA = energyDrinkA[0]
        prev_dpB = energyDrinkB[0]
        prev_dpC = -math.inf  # Represents an unreachable or non-beneficial state initially

        # Iterate from hour 1 up to n-1
        for i in range(1, n):
            # Calculate current hour's DP values based on previous hour's values

            # To calculate dpA for current hour (drinking A):
            # We either continued drinking A from the previous hour (prev_dpA)
            # or we just finished cleansing and started drinking A (prev_dpC).
            # Add the energy gained from A at the current hour.
            curr_dpA = max(prev_dpA, prev_dpC) + energyDrinkA[i]

            # To calculate dpB for current hour (drinking B):
            # We either continued drinking B from the previous hour (prev_dpB)
            # or we just finished cleansing and started drinking B (prev_dpC).
            # Add the energy gained from B at the current hour.
            curr_dpB = max(prev_dpB, prev_dpC) + energyDrinkB[i]

            # To calculate dpC for current hour (cleansing):
            # We gain 0 energy this hour.
            # We could have come from drinking A, drinking B, or already cleansing.
            # We take the maximum energy achieved before deciding to cleanse (or continue cleansing).
            curr_dpC = max(prev_dpA, prev_dpB, prev_dpC)

            # Update the 'previous' values for the next iteration
            prev_dpA = curr_dpA
            prev_dpB = curr_dpB
            prev_dpC = curr_dpC

        # The maximum total energy boost is the maximum of the three possible states
        # at the final hour (n-1).
        return max(prev_dpA, prev_dpB, prev_dpC)