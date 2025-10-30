class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        term1 = pow(26, n, MOD)
        term2 = (3 * pow(25, n, MOD)) % MOD
        term3 = (n * pow(25, n-1, MOD)) % MOD
        term4 = (3 * pow(24, n, MOD)) % MOD
        term5 = (2 * n * pow(24, n-1, MOD)) % MOD
        term6 = pow(23, n, MOD)
        term7 = (n * pow(23, n-1, MOD)) % MOD
        total = (term1 - term2 - term3 + term4 + term5 - term6 - term7) % MOD
        return total