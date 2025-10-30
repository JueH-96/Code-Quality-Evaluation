class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        k = (target - 1) // 2
        m = k + (1 if target % 2 == 0 else 0)
        if n <= m:
            total = n * (n + 1) // 2
        else:
            total = m * (m + 1) // 2
            rem = n - m
            sum_high = rem * (2 * target + rem - 1) // 2
            total += sum_high
        return total % MOD