class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        if n < 4:
            return 0
        
        pow26n = pow(26, n, MOD)
        pow25n = pow(25, n, MOD)
        pow25n_1 = pow(25, n-1, MOD)
        pow24n = pow(24, n, MOD)
        pow24n_1 = pow(24, n-1, MOD)
        pow23n = pow(23, n, MOD)
        pow23n_1 = pow(23, n-1, MOD)
        
        term1 = pow26n
        term2 = (3 * pow25n) % MOD
        term3 = (n * pow25n_1) % MOD
        term4 = (3 * pow24n) % MOD
        term5 = (2 * n * pow24n_1) % MOD
        term6 = pow23n
        term7 = (n * pow23n_1) % MOD
        
        good = (term1 - term2 - term3 + term4 + term5 - term6 - term7) % MOD
        return good