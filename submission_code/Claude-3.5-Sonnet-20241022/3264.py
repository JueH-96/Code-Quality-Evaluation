class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        n = len(enemyEnergies)
        
        # dp[i][j][k] represents max points possible when:
        # i: current index being considered
        # j: current energy
        # k: current points
        @cache
        def dp(i, energy, points):
            if i == n:
                return points
            
            # Skip current enemy
            result = dp(i + 1, energy, points)
            
            # Option 1: Defeat enemy if we have enough energy
            if energy >= enemyEnergies[i]:
                result = max(result, dp(i + 1, energy - enemyEnergies[i], points + 1))
            
            # Option 2: Mark enemy if we have at least 1 point
            if points > 0:
                result = max(result, dp(i + 1, energy + enemyEnergies[i], points))
                
            return result
            
        return dp(0, currentEnergy, 0)