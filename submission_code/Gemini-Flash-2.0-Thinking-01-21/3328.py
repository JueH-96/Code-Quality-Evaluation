import math

class Solution:
    def minOperations(self, k: int) -> int:
        if k <= 1:
            return 0
        min_operations = float('inf')
        upper_bound_x = int(2 * k**0.5) + 2
        for x in range(1, min(k, upper_bound_x) + 1):
            current_operations = (x - 1) + (math.ceil(k / x) - 1)
            min_operations = min(min_operations, current_operations)
        return min_operations