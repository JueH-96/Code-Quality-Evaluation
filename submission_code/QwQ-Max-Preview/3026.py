class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        m = min((target - 1) // 2, n)
        remaining = n - m
        sum_part1 = m * (m + 1) // 2
        sum_part2 = remaining * target + (remaining * (remaining - 1)) // 2
        return (sum_part1 + sum_part2) % MOD