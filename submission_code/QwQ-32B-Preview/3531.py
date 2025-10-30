class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        enemies = []
        
        for i in range(n):
            time_to_defeat = (health[i] + power - 1) // power
            ratio = damage[i] / time_to_defeat
            enemies.append((ratio, damage[i], time_to_defeat))
        
        # Sort enemies based on ratio descending
        enemies.sort(key=lambda x: -x[0])
        
        total_damage = 0
        remaining_sum = sum(damage)
        
        for enemy in enemies:
            time_j = enemy[2]
            damage_during_this_time = time_j * remaining_sum
            total_damage += damage_during_this_time
            remaining_sum -= enemy[1]
        
        return total_damage