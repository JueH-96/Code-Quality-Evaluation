class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        dp0 = energyDrinkA[0]
        dp1 = energyDrinkB[0]
        dp2 = -10**18
        
        for i in range(1, n):
            new_dp0 = max(dp0 + energyDrinkA[i], dp2 + energyDrinkA[i])
            new_dp1 = max(dp1 + energyDrinkB[i], dp2 + energyDrinkB[i])
            new_dp2 = max(dp0, dp1)
            dp0, dp1, dp2 = new_dp0, new_dp1, new_dp2
            
        return max(dp0, dp1, dp2)