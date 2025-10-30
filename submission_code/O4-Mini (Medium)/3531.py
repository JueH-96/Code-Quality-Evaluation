from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        # Compute time to kill each enemy
        times = [(damage[i], (health[i] + power - 1) // power) for i in range(n)]
        # Sort by descending ratio damage/time (Smith's rule for weighted sum of completion times)
        times.sort(key=lambda x: x[0] / x[1], reverse=True)
        
        total_damage = 0
        remaining_damage = sum(d for d, t in times)
        # Simulate killing in sorted order
        for d, t in times:
            total_damage += remaining_damage * t
            remaining_damage -= d
        
        return total_damage