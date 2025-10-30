from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        # Sort enemy energies in non-decreasing order.
        enemyEnergies.sort()
        left, right = 0, len(enemyEnergies) - 1
        points = 0
        max_points = 0
        
        # While there are enemies remaining:
        while left <= right:
            # If we can defeat the enemy with smallest energy, do so:
            if currentEnergy >= enemyEnergies[left]:
                currentEnergy -= enemyEnergies[left]
                points += 1
                left += 1
                max_points = max(max_points, points)
            # Otherwise, if we can use one of our points to gain energy by marking an enemy,
            # then sacrifice one point to recover energy.
            elif points > 0 and left < right:
                currentEnergy += enemyEnergies[right]
                points -= 1
                right -= 1
            else:
                break
        return max_points

# Sample test cases to validate.
if __name__ == '__main__':
    sol = Solution()
    
    # Example 1:
    enemyEnergies = [3,2,2]
    currentEnergy = 2
    print(sol.maximumPoints(enemyEnergies, currentEnergy))  # Expected output: 3
    
    # Example 2:
    enemyEnergies = [2]
    currentEnergy = 10
    print(sol.maximumPoints(enemyEnergies, currentEnergy))  # Expected output: 5