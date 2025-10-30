import math

class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        s0 = int(math.sqrt(k))
        min_sum = float('inf')
        # Check s from 1 to s0 + 1 inclusive
        for s in range(1, s0 + 2):
            x = (k + s - 1) // s  # Equivalent to ceil(k / s)
            current_sum = x + s
            if current_sum < min_sum:
                min_sum = current_sum
        return min_sum - 2