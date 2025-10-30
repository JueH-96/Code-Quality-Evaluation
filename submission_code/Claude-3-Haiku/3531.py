class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        total_damage = 0
        n = len(damage)
        
        while True:
            # Find the enemy that takes the least amount of time to kill
            min_time = float('inf')
            min_enemy = -1
            for i in range(n):
                if health[i] > 0:
                    time_to_kill = (health[i] + power - 1) // power
                    if time_to_kill < min_time:
                        min_time = time_to_kill
                        min_enemy = i
            
            # If all enemies are dead, break out of the loop
            if min_enemy == -1:
                break
            
            # Deal damage to the enemy and update the total damage
            total_damage += damage[min_enemy] * min_time
            health[min_enemy] -= power * min_time
        
        return total_damage