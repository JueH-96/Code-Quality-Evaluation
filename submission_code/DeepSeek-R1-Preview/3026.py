MOD = 10**9 + 7

class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        m = (target - 1) // 2
        if n <= m:
            return (n * (n + 1) // 2) % MOD
        else:
            sum_lower = m * (m + 1) // 2
            remaining = n - m
            sum_higher = remaining * (2 * target + (remaining - 1)) // 2
            total = sum_lower + sum_higher
            return total % MOD