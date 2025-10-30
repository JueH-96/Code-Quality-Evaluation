class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        min_energy = min(enemyEnergies)
        
        # If we can't defeat the weakest enemy, we can't get any points
        if currentEnergy < min_energy:
            return 0
        
        # Get the first point by defeating the weakest enemy
        points = 1
        currentEnergy -= min_energy
        
        # Absorb all other enemies to maximize energy
        total_energy_to_absorb = sum(enemyEnergies) - min_energy
        currentEnergy += total_energy_to_absorb
        
        # Use all remaining energy to defeat the weakest enemy
        points += currentEnergy // min_energy
        
        return points