class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        n = 1
        operations = 0
        while n * n < k:
            n += 1
            operations += 1
        return operations + (k - 1) // n