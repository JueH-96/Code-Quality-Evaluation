from typing import List
import math

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        # For each enemy, the number of seconds required to kill it is:
        #    t[i] = ceil(health[i] / power)
        # When fighting, every enemy still alive deals its damage each second.
        #
        # If we fix an order in which Bob kills the enemies, then an enemy i that is killed at
        # time T (i.e. after T rounds) contributes damage[i] points in each round until it dies,
        # which gives a cost of T * damage[i].
        # In an optimal schedule, we wish to minimize the total received damage:
        #
        #   total_damage = sum (T_i * damage[i])
        #
        # where T_i is the round when enemy i is killed. However, Bob can choose the order in
        # which to bring down the enemies.
        #
        # Note that the total number of rounds is fixed: T = sum(t[i]). But which enemy dies
        # earlier (has a smaller T_i) is crucial to reducing the overall damage – because as long as
        # an enemy is alive, it contributes.
        #
        # We can view this as a scheduling problem where:
        #   - Each enemy i has a "processing time" t[i] (the time required to kill it)
        #   - And a "weight" damage[i] (the penalty per second it remains alive)
        #
        # The total damage is equivalent to the sum of weighted completion times.
        # The well known “Smith’s rule” for minimizing the weighted sum of completion times is to 
        # schedule jobs in order of non-decreasing ratio (processing_time / weight).
        #
        # For our enemies, we have:
        #   t[i] = ceil(health[i] / power)   and   weight = damage[i]
        # Thus, it is optimal to fight the enemies in an order sorted by:
        #   key = t[i] / damage[i]
        #
        # Since this ratio is rational and our numbers are small, we can simply sort
        # using the float value (or do a cross multiplication comparison).
        #
        # Once sorted, we compute the cumulative rounds and add the product (cumulative rounds * damage)
        # for each enemy.
        n = len(damage)
        # Create a list of tuples with (t[i], damage[i])
        stats = []
        for i in range(n):
            t = (health[i] + power - 1) // power  # number of seconds required to kill enemy i
            stats.append((t, damage[i]))
        
        # Sort the enemies by the ratio t/damage (Smith's rule)
        stats.sort(key=lambda x: x[0] / x[1])
        
        total_damage = 0
        cumulative_time = 0
        for t, d in stats:
            cumulative_time += t
            total_damage += cumulative_time * d
        return total_damage


# For example, you can run the following tests:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1
    power = 4
    damage = [1, 2, 3, 4]
    health = [4, 5, 6, 8]
    print(sol.minDamage(power, damage, health))  # Expected output 39

    # Example 2
    power = 1
    damage = [1, 1, 1, 1]
    health = [1, 2, 3, 4]
    print(sol.minDamage(power, damage, health))  # Expected output 20

    # Example 3
    power = 8
    damage = [40]
    health = [59]
    print(sol.minDamage(power, damage, health))  # Expected output 320