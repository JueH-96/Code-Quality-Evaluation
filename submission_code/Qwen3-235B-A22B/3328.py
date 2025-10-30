import math

class Solution:
    def minOperations(self, k: int) -> int:
        min_ops = float('inf')
        sqrt_k = math.isqrt(k)
        for x in range(1, sqrt_k + 2):
            m = (k + x - 1) // x  # Calculate ceil(k / x)
            current_sum = x + m
            if current_sum < min_ops:
                min_ops = current_sum
        return min_ops - 2