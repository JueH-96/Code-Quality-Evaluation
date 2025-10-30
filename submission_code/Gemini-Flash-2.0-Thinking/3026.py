class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7

        def sum_arithmetic_progression(start, count):
            return (count * start + count * (count - 1) // 2) % MOD

        if target % 2 == 1:
            k = target // 2
            if n <= k:
                return (n * (n + 1) // 2) % MOD
            else:
                sum1 = (k * (k + 1) // 2) % MOD
                remaining = n - k
                sum2_start = target
                sum2 = sum_arithmetic_progression(sum2_start, remaining)
                return (sum1 + sum2) % MOD
        else:
            k = target // 2
            if n <= k - 1:
                return (n * (n + 1) // 2) % MOD
            elif n == k:
                return (k * (k + 1) // 2) % MOD
            else:
                sum1 = (k * (k + 1) // 2) % MOD
                remaining = n - k
                sum2_start = target
                sum2 = sum_arithmetic_progression(sum2_start, remaining)
                return (sum1 + sum2) % MOD