import math

class Solution:
    def minOperations(self, k: int) -> int:
        if k <= 1:
            return 0

        min_ops = float('inf')
        for num_duplicates in range(k):
            num_elements = num_duplicates + 1
            target_value = math.ceil(k / num_elements)
            num_increases = target_value - 1
            total_operations = num_duplicates + num_increases
            min_ops = min(min_ops, total_operations)

        return min_ops