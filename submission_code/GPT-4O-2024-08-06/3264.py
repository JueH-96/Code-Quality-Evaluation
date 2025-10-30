class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        # Sort the enemy energies to prioritize defeating weaker enemies first
        enemyEnergies.sort()
        
        points = 0
        i, j = 0, len(enemyEnergies) - 1
        
        while i <= j:
            # If we can defeat the current weakest enemy
            if currentEnergy >= enemyEnergies[i]:
                currentEnergy -= enemyEnergies[i]
                points += 1
                i += 1
            # If we have at least one point, we can mark the strongest unmarked enemy
            elif points > 0:
                currentEnergy += enemyEnergies[j]
                points -= 1
                j -= 1
            else:
                # If we can't defeat any enemy and can't mark any enemy, break the loop
                break
        
        return points