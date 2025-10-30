class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        if n == 1:
            return 1
        if target == 1:
            return (n * (n + 1) // 2) % MOD
        if target == 2:
            return ((n - 1) * n // 2 + n) % MOD
        if target > n:
            return ((n - 1) * n // 2 + n) % MOD
        if target <= n // 2:
            return ((target - 1) * target // 2 + target) % MOD
        return ((n - target + 1) * (n - target + 2) // 2 + (target - 1) * target // 2) % MOD