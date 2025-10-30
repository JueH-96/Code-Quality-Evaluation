class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        n = 1
        operations = 0
        while n < k:
            if n * 2 < k:
                n *= 2
                operations += 1
            else:
                n += 1
                operations += 1
        return operations