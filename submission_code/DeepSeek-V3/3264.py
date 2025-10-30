class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        # Sort the enemy energies to prioritize smaller ones first
        enemyEnergies.sort()
        points = 0
        marked = [False] * len(enemyEnergies)
        
        # First, try to gain as many points as possible by defeating enemies
        for i in range(len(enemyEnergies)):
            if currentEnergy >= enemyEnergies[i]:
                points += 1
                currentEnergy -= enemyEnergies[i]
                marked[i] = True
        
        # Now, try to regain energy by marking enemies
        # We need to mark the largest possible enemies first to maximize energy gain
        # So, we iterate from the end
        for i in range(len(enemyEnergies) - 1, -1, -1):
            if not marked[i] and points >= 1:
                currentEnergy += enemyEnergies[i]
                marked[i] = True
                # After marking, try to defeat more enemies
                for j in range(len(enemyEnergies)):
                    if not marked[j] and currentEnergy >= enemyEnergies[j]:
                        points += 1
                        currentEnergy -= enemyEnergies[j]
                        marked[j] = True
        
        return points