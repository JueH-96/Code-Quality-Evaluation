MOD = 10**9 + 7

class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        m = target // 2
        if n <= m:
            return (n * (n + 1) // 2) % MOD
        else:
            sum_lower = m * (m + 1) // 2
            sum_upper = (n - m) * target + ((n - m - 1) * (n - m)) // 2
            total = sum_lower + sum_upper
            return total % MOD