class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        if n < 4:
            return 0
        pow26n = pow(26, n, MOD)
        term_a = (3 * pow(25, n, MOD)) % MOD
        term_b = (n * pow(25, n-1, MOD)) % MOD
        term_c = (3 * pow(24, n, MOD)) % MOD
        term_d = (2 * n * pow(24, n-1, MOD)) % MOD
        term_e = pow(23, n, MOD)
        term_f = (n * pow(23, n-1, MOD)) % MOD
        
        res = (pow26n - term_a - term_b) % MOD
        res = (res + term_c + term_d) % MOD
        res = (res - term_e - term_f) % MOD
        return res % MOD