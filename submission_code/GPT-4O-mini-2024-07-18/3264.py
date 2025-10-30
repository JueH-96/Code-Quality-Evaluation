class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        points = 0
        n = len(enemyEnergies)
        
        # First, we try to gain points by defeating enemies
        for energy in enemyEnergies:
            if currentEnergy >= energy:
                points += 1
                currentEnergy -= energy
            else:
                break
        
        # Now we can mark enemies and regain energy
        # We can mark enemies from the beginning to the end
        for i in range(points):
            currentEnergy += enemyEnergies[i]
        
        # After marking, we can again try to defeat enemies
        for energy in enemyEnergies[points:]:
            while currentEnergy >= energy:
                points += 1
                currentEnergy -= energy
        
        return points