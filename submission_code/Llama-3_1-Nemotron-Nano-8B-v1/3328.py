import math

class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        min_steps = float('inf')
        max_m = int(math.sqrt(k)) + 2
        for m in range(1, max_m + 1):
            t_plus_1 = (k + m - 1) // m  # Equivalent to ceil(k/m)
            steps = (m - 1) + (t_plus_1 - 1)
            if steps < min_steps:
                min_steps = steps
        return min_steps