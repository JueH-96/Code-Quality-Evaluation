from typing import List
import heapq

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        # Pair each enemy's damage with their health and sort by damage in descending order
        enemies = sorted(zip(damage, health), key=lambda x: -x[0])

        total_damage = 0
        current_health = [h for d, h in enemies]

        while any(h > 0 for h in current_health):
            # Calculate the total damage dealt to Bob this second
            damage_this_second = sum(d for d, h in zip(damage, current_health) if h > 0)
            total_damage += damage_this_second

            # Find the enemy with the highest damage output that is still alive
            for i in range(len(enemies)):
                if current_health[i] > 0:
                    current_health[i] -= power
                    if current_health[i] < 0:
                        current_health[i] = 0
                    break

        return total_damage