class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        enemies = sorted(zip(damage, health), key=lambda x: x[0], reverse=True)
        total_damage = 0
        
        while enemies:
            current_damage = sum(d for d, h in enemies)
            total_damage += current_damage
            
            time_to_kill = (enemies[-1][1] + power - 1) // power
            total_damage += (time_to_kill - 1) * current_damage
            
            enemies = [(d, h - time_to_kill * power) for d, h in enemies if h > time_to_kill * power]
        
        return total_damage