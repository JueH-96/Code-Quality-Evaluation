class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # Initialize the maximum energy boost to 0
        max_energy_boost = 0
        
        # Try starting with energy drink A
        energy_boost = 0
        for i in range(n):
            energy_boost += energyDrinkA[i]
            if i < n - 1:
                energy_boost = max(energy_boost, energy_boost + energyDrinkB[i+1])
            max_energy_boost = max(max_energy_boost, energy_boost)
        
        # Try starting with energy drink B
        energy_boost = 0
        for i in range(n):
            energy_boost += energyDrinkB[i]
            if i < n - 1:
                energy_boost = max(energy_boost, energy_boost + energyDrinkA[i+1])
            max_energy_boost = max(max_energy_boost, energy_boost)
        
        return max_energy_boost