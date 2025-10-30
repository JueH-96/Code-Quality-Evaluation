class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        if n == 0:
            return 0
        dp_a = energyDrinkA[0]
        dp_b = energyDrinkB[0]
        for i in range(1, n):
            next_dp_a = max(dp_a + energyDrinkA[i], dp_b)
            next_dp_b = max(dp_b + energyDrinkB[i], dp_a)
            dp_a = next_dp_a
            dp_b = next_dp_b
        return max(dp_a, dp_b)