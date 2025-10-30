class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        # Number of hits needed to kill enemy i
        # T_i = ceil(health[i] / power) = (health[i] + power - 1) // power
        hits_needed = [(health[i] + power - 1) // power for i in range(n)]
        
        # Sort enemies in descending order of damage[i] / hits_needed[i]
        enemies = list(range(n))
        enemies.sort(key=lambda i: damage[i] / hits_needed[i], reverse=True)
        
        total_damage = 0
        current_time = 0  # This will track the completion time of each enemy
        for i in enemies:
            current_time += hits_needed[i]
            total_damage += damage[i] * current_time
        
        return total_damage