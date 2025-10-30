class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        if currentEnergy == 0:
            return 0
        e_min = min(enemyEnergies)
        if currentEnergy < e_min:
            return 0
        return (currentEnergy // e_min) + (len(enemyEnergies) - 1)