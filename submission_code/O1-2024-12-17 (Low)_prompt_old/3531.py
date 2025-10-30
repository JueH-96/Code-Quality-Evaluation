class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        import math

        n = len(damage)
        # Calculate how many hits (seconds of attacking) each enemy needs:
        # hits[i] = ceil(health[i] / power)
        hits = [(health[i] + power - 1) // power for i in range(n)]
        
        # Pair each enemy with (damage, hits) and sort in descending order of damage/hits
        enemies = [(damage[i], hits[i]) for i in range(n)]
        enemies.sort(key=lambda x: x[0] / x[1], reverse=True)

        total_damage = 0
        elapsed_hits = 0  # sum of hits so far

        # After sorting, compute total damage
        for d, h in enemies:
            elapsed_hits += h
            total_damage += d * elapsed_hits

        return total_damage