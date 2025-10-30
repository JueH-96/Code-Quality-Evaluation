from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        # Initialize dp for hour 0: [drink A, drink B, idle]
        prev_dp = [energyDrinkA[0], energyDrinkB[0], 0]
        
        # Iterate from hour 1 to n-1
        for i in range(1, n):
            # Current dp values
            curr_dp = [
                max(prev_dp[0], prev_dp[2]) + energyDrinkA[i],  # Drink A: can come from A or idle
                max(prev_dp[1], prev_dp[2]) + energyDrinkB[i],  # Drink B: can come from B or idle
                max(prev_dp[0], prev_dp[1], prev_dp[2])         # Idle: can come from any state, no boost
            ]
            # Update previous dp to current dp for next iteration
            prev_dp = curr_dp
        
        # The answer is the maximum of the three states at hour n-1
        return max(prev_dp)