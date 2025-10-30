class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        min_energy = min(enemyEnergies)
        
        # If we can't defeat the weakest enemy, we can't get any points
        if currentEnergy < min_energy:
            return 0
        
        # Calculate total energy we can get by absorbing all enemies except the weakest one
        total_energy = currentEnergy + sum(enemyEnergies) - min_energy
        
        # Maximum points = total available energy divided by minimum energy cost
        return total_energy // min_energy