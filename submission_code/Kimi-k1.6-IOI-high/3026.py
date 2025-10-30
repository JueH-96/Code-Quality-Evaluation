class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        m = target // 2
        if m >= n:
            return (n * (n + 1) // 2) % MOD
        else:
            k = n - m
            sum1 = m * (m + 1) // 2
            sum2 = k * (2 * target + (k - 1)) // 2
            return (sum1 + sum2) % MOD