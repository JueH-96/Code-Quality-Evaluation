class Solution:
    def minOperations(self, k: int) -> int:
        ans = float('inf')
        for n in range(1, k + 1):
            if n - 1 >= ans:
                break
            M = (k + n - 1) // n
            operations = M + n - 2
            if operations < ans:
                ans = operations
        return ans