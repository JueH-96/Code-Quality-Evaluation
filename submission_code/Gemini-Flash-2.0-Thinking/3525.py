class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        if n == 0:
            return 0

        if n == 1:
            return max(energyDrinkA[0], energyDrinkB[0])

        dpA_prev = energyDrinkA[0]
        dpB_prev = energyDrinkB[0]

        if n == 2:
            dpA_curr = max(dpA_prev + energyDrinkA[1], energyDrinkA[1])
            dpB_curr = max(dpB_prev + energyDrinkB[1], energyDrinkB[1])
            return max(dpA_curr, dpB_curr)

        dpA_curr = max(dpA_prev + energyDrinkA[1], energyDrinkA[1])
        dpB_curr = max(dpB_prev + energyDrinkB[1], energyDrinkB[1])

        dpA_prev_prev = dpA_prev
        dpB_prev_prev = dpB_prev
        dpA_prev = dpA_curr
        dpB_prev = dpB_curr

        for i in range(2, n):
            new_dpA_curr = max(dpA_prev + energyDrinkA[i], dpB_prev_prev + energyDrinkA[i])
            new_dpB_curr = max(dpB_prev + energyDrinkB[i], dpA_prev_prev + energyDrinkB[i])

            dpA_prev_prev = dpA_prev
            dpB_prev_prev = dpB_prev
            dpA_prev = new_dpA_curr
            dpB_prev = new_dpB_curr

        return max(dpA_prev, dpB_prev)