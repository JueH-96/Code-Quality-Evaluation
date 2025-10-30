from functools import lru_cache
from math import log2

class Solution:
    def waysToReachStair(self, k: int) -> int:
        if k == 0:
            return 2  # Base case as per the problem statement

        @lru_cache(maxsize=None)
        def dfs(steps):
            if steps == 1:
                return 4  # Base case from the problem statement and example 2
            if steps <= 0:
                return 0

            # Calculate the number of steps back to the nearest power of 2 jump
            closest_pow_2_step = int(log2(steps - 1)) if steps > 1 else 0
            # The total ways are the sum of two cases:
            # 1. Directly jumping (2^j) to the current step (steps) from somewhere (steps - 2^j).
            # 2. Coming from one step lower (steps - 1), which follows the down-to-up constraint.
            total_ways = dfs(steps - 2**closest_pow_2_step) + dfs(steps - 1)
            return total_ways

        return dfs(k)