import math

class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        m = math.isqrt(k) + 1
        min_sum = float('inf')
        for x in range(1, m + 1):
            y = (k + x - 1) // x  # Equivalent to ceiling(k / x)
            current_sum = x + y
            if current_sum < min_sum:
                min_sum = current_sum
        return min_sum - 2