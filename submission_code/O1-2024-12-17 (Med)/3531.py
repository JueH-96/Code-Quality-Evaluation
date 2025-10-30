class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        
        # Calculate how many hits ("processing time") each enemy needs
        # hits[i] = number of hits needed to kill enemy i
        hits = [(h + power - 1) // power for h in health]
        
        # Pair each enemy with (damage, hits) and calculate ratio = damage / hits
        enemies = []
        for i in range(n):
            enemies.append((damage[i], hits[i], damage[i] / hits[i]))
        
        # Sort enemies by descending ratio (damage[i] / hits[i])
        enemies.sort(key=lambda x: x[2], reverse=True)
        
        total_damage = 0
        current_time = 0
        
        # Schedule enemies in the sorted order; each second we deliver one hit
        # The "completion time" for each enemy i is current_time + hits[i],
        # and that enemy contributes damage[i] * (completion time).
        for dmg, h, _ in enemies:
            current_time += h
            total_damage += dmg * current_time
        
        return total_damage