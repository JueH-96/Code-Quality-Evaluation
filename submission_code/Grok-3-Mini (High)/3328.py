class Solution:
    def minOperations(self, k: int) -> int:
        min_ops = k - 1  # Initialize with an upper bound achievable when V = k
        for val in range(1, k + 1):
            ceil_div = (k + val - 1) // val  # Compute ceil(k / val) using integer arithmetic
            ops = val + ceil_div - 2
            if ops < min_ops:
                min_ops = ops
        return min_ops