class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        points = 0
        i = 0
        n = len(enemyEnergies)
        
        # First pass: try to defeat as many enemies as possible
        while i < n and currentEnergy >= enemyEnergies[i]:
            currentEnergy -= enemyEnergies[i]
            points += 1
            i += 1
        
        # Second pass: use points to absorb energies of remaining enemies
        for j in range(i, n):
            if points > 0:
                currentEnergy += enemyEnergies[j]
                points -= 1
            else:
                break
        
        # Third pass: try to defeat again with the new energy
        while i < n and currentEnergy >= enemyEnergies[i]:
            currentEnergy -= enemyEnergies[i]
            points += 1
            i += 1
        
        return points