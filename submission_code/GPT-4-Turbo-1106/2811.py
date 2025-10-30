class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        if n == 1:
            return 1
        if k % 2 == 0 and n > k // 2:
            return (k - 1) * n // 2 + 1
        return (k - 1) * n // 2