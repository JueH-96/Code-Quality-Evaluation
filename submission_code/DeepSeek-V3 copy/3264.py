class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        # Sort the enemy energies in ascending order
        enemyEnergies.sort()
        points = 0
        i = 0
        n = len(enemyEnergies)
        
        # First, try to gain as many points as possible by defeating enemies
        while i < n and currentEnergy >= enemyEnergies[i]:
            currentEnergy -= enemyEnergies[i]
            points += 1
            i += 1
        
        # Now, try to use the marked enemies to gain more points
        # We need to have at least one point to start marking enemies
        if points > 0:
            # Iterate from the end to the start to maximize the energy gain
            j = n - 1
            while j >= i:
                if currentEnergy >= enemyEnergies[j]:
                    # Defeat the enemy to gain a point
                    currentEnergy -= enemyEnergies[j]
                    points += 1
                    j -= 1
                else:
                    # Mark the enemy to gain energy
                    currentEnergy += enemyEnergies[j]
                    j -= 1
        
        return points