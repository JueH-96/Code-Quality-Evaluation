class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 1000000007
        max_small = target // 2
        if n <= max_small:
            return (n * (n + 1) // 2) % MOD
        else:
            sum_small = (max_small * (max_small + 1) // 2)
            m = n - max_small
            sum_add = (m * target) + (m * (m - 1) // 2)
            total_sum = (sum_small + sum_add) % MOD
            return total_sum