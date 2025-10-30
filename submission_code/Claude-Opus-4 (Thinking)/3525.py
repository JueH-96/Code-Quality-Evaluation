class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # dp_a[i] = max energy boost up to hour i, ending with drinking A at hour i
        # dp_b[i] = max energy boost up to hour i, ending with drinking B at hour i
        dp_a = [0] * n
        dp_b = [0] * n
        
        # Base case: start with either drink
        dp_a[0] = energyDrinkA[0]
        dp_b[0] = energyDrinkB[0]
        
        for i in range(1, n):
            # To drink A at hour i
            dp_a[i] = dp_a[i-1] + energyDrinkA[i]  # Continue drinking A
            if i >= 2:
                dp_a[i] = max(dp_a[i], dp_b[i-2] + energyDrinkA[i])  # Switch from B (waited at hour i-1)
            
            # To drink B at hour i
            dp_b[i] = dp_b[i-1] + energyDrinkB[i]  # Continue drinking B
            if i >= 2:
                dp_b[i] = max(dp_b[i], dp_a[i-2] + energyDrinkB[i])  # Switch from A (waited at hour i-1)
        
        return max(dp_a[n-1], dp_b[n-1])