class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        inv2 = pow(2, MOD - 2, MOD)
        k = (target - 1) // 2
        if n <= k:
            result = (n * (n + 1) % MOD * inv2) % MOD
        else:
            m = n - k
            sum1 = (k * (k + 1) % MOD * inv2) % MOD
            sum2 = ((m - 1) * m % MOD * inv2) % MOD
            sum3 = (m * target) % MOD
            result = (sum1 + sum2 + sum3) % MOD
        return result