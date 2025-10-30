class Solution:
    def maximumPoints(self, enemyEnergies: list[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        points = 0
        stack = []
        while enemyEnergies or stack:
            if enemyEnergies and currentEnergy >= enemyEnergies[0]:
                currentEnergy -= enemyEnergies[0]
                points += 1
                stack.append(enemyEnergies.pop(0))
            elif stack:
                currentEnergy += stack.pop()
            else:
                break
        return points