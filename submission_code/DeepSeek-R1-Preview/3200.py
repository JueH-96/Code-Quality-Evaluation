class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        a = pow(26, n, MOD)
        b = (3 * pow(25, n, MOD)) % MOD
        c = (n * pow(25, n - 1, MOD)) % MOD
        d = (3 * pow(24, n, MOD)) % MOD
        e = (2 * n * pow(24, n - 1, MOD)) % MOD
        f = pow(23, n, MOD)
        g = (n * pow(23, n - 1, MOD)) % MOD
        
        res = (a - b - c + d + e - f - g) % MOD
        return res