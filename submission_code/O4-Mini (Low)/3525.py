from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        if n == 0:
            return 0
        if n == 1:
            return max(energyDrinkA[0], energyDrinkB[0])
        # dpA[i]: max boost if we drink A at hour i
        # dpB[i]: max boost if we drink B at hour i
        # Transitions:
        #   Continue same drink: dpA[i-1] + A[i], dpB[i-1] + B[i]
        #   Switch drink: must skip exactly one hour, so
        #     dpA[i] from dpB[i-2] + A[i]
        #     dpB[i] from dpA[i-2] + B[i]
        # Initialize for i=0,1:
        dpA_prev2 = energyDrinkA[0]  # dpA[0]
        dpB_prev2 = energyDrinkB[0]  # dpB[0]
        # i = 1
        dpA_prev1 = dpA_prev2 + energyDrinkA[1]
        dpB_prev1 = dpB_prev2 + energyDrinkB[1]
        # iterate from i = 2 .. n-1
        for i in range(2, n):
            a_i = energyDrinkA[i]
            b_i = energyDrinkB[i]
            # continue same
            contA = dpA_prev1 + a_i
            contB = dpB_prev1 + b_i
            # switch (skip exactly one hour)
            switchA = dpB_prev2 + a_i
            switchB = dpA_prev2 + b_i
            dpA_i = max(contA, switchA)
            dpB_i = max(contB, switchB)
            # shift for next
            dpA_prev2, dpA_prev1 = dpA_prev1, dpA_i
            dpB_prev2, dpB_prev1 = dpB_prev1, dpB_i
        # result is max if we end by drinking A or B at last hour
        return max(dpA_prev1, dpB_prev1)