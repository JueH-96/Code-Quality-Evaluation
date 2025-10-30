class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        min_energy = min(enemyEnergies)
        
        # If we can't defeat even the weakest enemy, we get 0 points
        if currentEnergy < min_energy:
            return 0
        
        # If there's only one enemy, we can only repeatedly defeat it
        if len(enemyEnergies) == 1:
            return currentEnergy // min_energy
        
        # For multiple enemies: we'll mark all except the weakest one
        # Total energy = initial + sum of all enemies - min (the one we keep unmarked)
        total_energy = currentEnergy + sum(enemyEnergies) - min_energy
        
        # We can defeat the weakest enemy this many times
        return total_energy // min_energy