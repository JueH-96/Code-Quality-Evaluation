class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        dp_a = [0] * n
        dp_b = [0] * n

        dp_a[0] = energyDrinkA[0]
        dp_b[0] = energyDrinkB[0]

        for i in range(1, n):
            dp_a[i] = max(dp_a[i - 1] + energyDrinkA[i], dp_b[i - 2] + energyDrinkA[i] if i >= 2 else dp_b[0])
            dp_b[i] = max(dp_b[i - 1] + energyDrinkB[i], dp_a[i - 2] + energyDrinkB[i] if i >= 2 else dp_a[0])

        return max(dp_a[n - 1], dp_b[n - 1])