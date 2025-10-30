class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        min_energy = min(enemyEnergies)
        
        # If we can't defeat the weakest enemy, we can't get any points
        if currentEnergy < min_energy:
            return 0
        
        # Total energy available = current energy + sum of all enemy energies
        total_energy = currentEnergy + sum(enemyEnergies)
        
        # We can repeatedly defeat the weakest enemy
        return total_energy // min_energy