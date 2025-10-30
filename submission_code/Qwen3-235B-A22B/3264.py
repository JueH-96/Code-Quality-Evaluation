class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        n = len(enemyEnergies)
        left = 0
        right = n - 1
        points = 0
        
        while True:
            # Try to attack the leftmost possible enemy
            if currentEnergy >= enemyEnergies[left]:
                currentEnergy -= enemyEnergies[left]
                points += 1
                left += 1
            else:
                # Can't attack, try to collect from right
                if points > 0 and right >= left:
                    currentEnergy += enemyEnergies[right]
                    right -= 1
                else:
                    break
            # After collecting, continue loop to attack again if possible
            # Exit condition: if left exceeds right or no progress can be made
            if left > right:
                break
        
        return points