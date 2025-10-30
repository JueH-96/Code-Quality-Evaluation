class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        n = len(enemyEnergies)
        points = 0
        marked = [False] * n
        
        def dfs(index, energy):
            nonlocal points
            if index == n:
                return
            
            # Try to defeat the enemy
            if energy >= enemyEnergies[index]:
                points += 1
                dfs(index + 1, energy - enemyEnergies[index])
                points -= 1
            
            # Try to mark the enemy
            if points > 0 and not marked[index]:
                points -= 1
                marked[index] = True
                dfs(index + 1, energy + enemyEnergies[index])
                marked[index] = False
                points += 1
            
            # Skip this enemy
            dfs(index + 1, energy)
        
        dfs(0, currentEnergy)
        return points