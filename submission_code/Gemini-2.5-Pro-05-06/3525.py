import math
from typing import List

class Solution:
  def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
    n = len(energyDrinkA)

    # dp_drinking_A: max energy ending at current hour, consuming A this hour.
    # dp_drinking_B: max energy ending at current hour, consuming B this hour.
    # dp_cleansing_for_A: max energy ending at current hour, if cleansing this hour to prepare for A.
    #                     This means B was consumed at the previous hour. Energy gain this hour is 0.
    # dp_cleansing_for_B: max energy ending at current hour, if cleansing this hour to prepare for B.
    #                     This means A was consumed at the previous hour. Energy gain this hour is 0.

    # Base case: hour 0 (index 0 in arrays)
    # Initialize 'prev_' states which will hold DP values for hour i-1.
    # For the first iteration (i=1), 'prev_' states are effectively dp[0] values.
    
    # At hour 0, we start by drinking one of the drinks. No prior cleansing needed.
    # Using float for sums initially because -math.inf is a float.
    prev_drinking_A = float(energyDrinkA[0])
    prev_drinking_B = float(energyDrinkB[0])
    
    # It's not possible to be in a cleansing state at hour 0, because cleansing implies
    # having consumed a *different* type of drink in the hour before cleansing.
    # Initialize these to negative infinity to ensure they are not chosen as a path origin.
    prev_cleansing_for_A = -math.inf 
    prev_cleansing_for_B = -math.inf

    # Iterate from hour 1 (index 1) up to n-1 (index n-1)
    for i in range(1, n):
        # Calculate DP states for the current hour 'i' based on 'prev_' states (for hour i-1).
        
        # Max energy if consuming drink A at hour 'i':
        # Can come from:
        # 1. Consuming A at hour i-1: prev_drinking_A
        # 2. Cleansing for A at hour i-1 (i.e., finished cleansing, ready for A): prev_cleansing_for_A
        # Add energyDrinkA[i] to the chosen maximum from previous states.
        curr_drinking_A = max(prev_drinking_A, prev_cleansing_for_A) + energyDrinkA[i]
        
        # Max energy if consuming drink B at hour 'i': (Similar logic)
        curr_drinking_B = max(prev_drinking_B, prev_cleansing_for_B) + energyDrinkB[i]
        
        # Max energy if cleansing at hour 'i' to prepare for drink A (next hour):
        # This means drink B was consumed at hour i-1.
        # Energy gain at hour 'i' is 0 (due to cleansing).
        # Total accumulated energy is same as prev_drinking_B (energy up to end of hour i-1).
        curr_cleansing_for_A = prev_drinking_B
        
        # Max energy if cleansing at hour 'i' to prepare for drink B (next hour):
        # This means drink A was consumed at hour i-1. (Similar logic)
        curr_cleansing_for_B = prev_drinking_A
        
        # Update 'prev_' states to current states for the next iteration (i+1).
        prev_drinking_A = curr_drinking_A
        prev_drinking_B = curr_drinking_B
        prev_cleansing_for_A = curr_cleansing_for_A
        prev_cleansing_for_B = curr_cleansing_for_B
            
    # After the loop, 'prev_' variables hold the DP values for the last hour (n-1).
    # The maximum total energy boost is the maximum among all possible states at hour n-1.
    max_energy = max(prev_drinking_A, prev_drinking_B, prev_cleansing_for_A, prev_cleansing_for_B)
    
    # Since all energyDrink values are positive integers, and n >= 3 (problem constraint),
    # all DP states (prev_drinking_A, etc.) will be finite sums of integers by the end.
    # -math.inf is only an initial placeholder.
    # The result will be a float that represents an integer, so convert to int.
    return int(max_energy)