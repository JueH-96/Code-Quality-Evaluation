class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        m = target // 2
        if n <= m:
            return (n * (n + 1) // 2) % MOD
        else:
            rem = n - m
            sum_lower = m * (m + 1) // 2
            sum_upper = rem * target + rem * (rem - 1) // 2
            total = sum_lower + sum_upper
            return total % MOD