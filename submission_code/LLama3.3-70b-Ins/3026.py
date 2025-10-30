class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        if target % 2 == 0:
            return (n * (n + 1) // 2 + n * (n - 1) // 2) % MOD
        else:
            return (n * (n + 1) // 2) % MOD