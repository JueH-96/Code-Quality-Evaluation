from typing import List
import functools

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        # Compute the number of hits (seconds) needed to kill each enemy
        # t_i = ceil(health[i] / power)
        jobs = []
        for d, h in zip(damage, health):
            t = (h + power - 1) // power
            jobs.append((d, t))

        # We need to schedule jobs (enemies) to minimize sum(w_i * C_i),
        # where w_i = damage[i] and C_i = completion time (in seconds).
        # By Smith's rule, sort by descending ratio w_i / t_i.
        def cmp(a, b):
            # Compare a[0]/a[1] and b[0]/b[1] via cross‐multiplication
            lhs = a[0] * b[1]
            rhs = b[0] * a[1]
            if lhs > rhs:
                return -1  # a has higher ratio, comes first
            if lhs < rhs:
                return 1
            # Tie‐break: kill the one with smaller processing time first
            if a[1] < b[1]:
                return -1
            if a[1] > b[1]:
                return 1
            return 0

        jobs.sort(key=functools.cmp_to_key(cmp))

        # Now accumulate total damage = sum over jobs of w_i * (sum of t up to that job)
        total_damage = 0
        current_time = 0
        for w, t in jobs:
            current_time += t
            total_damage += w * current_time

        return total_damage