from typing import List
import math

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        # For each enemy, compute rounds required:
        # rounds = ceil(health[i] / power)
        enemies = []
        for i in range(n):
            rounds = (health[i] + power - 1) // power
            enemies.append((rounds, damage[i]))
        
        # To minimize total weighted completion time, we order enemies by increasing p/w = rounds/damage.
        # Instead of computing float division (which is safe given constraints), to be exact,
        # we can sort using a key that compares rounds/damage by cross multiplication.
        # A lambda using float division is acceptable.
        enemies.sort(key=lambda x: x[0] / x[1])
        
        total_damage = 0
        current_time = 0
        # In this schedule, each enemy i is killed at time = current_time + rounds[i].
        for rounds, d in enemies:
            current_time += rounds
            total_damage += d * current_time
        return total_damage

# For local testing purpose:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    power = 4
    damage = [1,2,3,4]
    health = [4,5,6,8]
    print(sol.minDamage(power, damage, health))  # Expected 39
    
    # Example 2:
    power = 1
    damage = [1,1,1,1]
    health = [1,2,3,4]
    print(sol.minDamage(power, damage, health))  # Expected 20
    
    # Example 3:
    power = 8
    damage = [40]
    health = [59]
    print(sol.minDamage(power, damage, health))  # Expected 320