class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        n = len(enemyEnergies)
        left, right = 0, n - 1
        points = 0
        while left <= right:
            if currentEnergy >= enemyEnergies[left]:
                currentEnergy -= enemyEnergies[left]
                points += 1
                left += 1
            elif points > 0 and left < right:
                currentEnergy += enemyEnergies[right]
                points -= 1
                right -= 1
            else:
                break
        return points