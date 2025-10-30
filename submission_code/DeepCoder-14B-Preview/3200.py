class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        if n < 4:
            return 0
        pow_26 = pow(26, n, MOD)
        t2 = (3 * pow(25, n, MOD)) % MOD
        t3 = (n * pow(25, n-1, MOD)) % MOD
        t4 = (3 * pow(24, n, MOD)) % MOD
        t5 = (2 * n * pow(24, n-1, MOD)) % MOD
        t6 = pow(23, n, MOD) % MOD
        t7 = (n * pow(23, n-1, MOD)) % MOD
        res = (pow_26 - t2 - t3 + t4 + t5 - t6 - t7) % MOD
        return res