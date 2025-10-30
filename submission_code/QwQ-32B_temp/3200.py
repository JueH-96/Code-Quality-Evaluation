class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        if n < 4:
            return 0
        
        term1 = pow(26, n, MOD)
        
        # Compute terms for |A| + |B| + |C|
        termA = 3 * pow(25, n, MOD) % MOD
        termB = n * pow(25, n-1, MOD) % MOD
        sum_term2 = (termA + termB) % MOD
        
        # Compute terms for intersections of two sets (A∩B, A∩C, B∩C)
        termC = 3 * pow(24, n, MOD) % MOD
        termD = (2 * n * pow(24, n-1, MOD)) % MOD
        sum_term4 = (termC + termD) % MOD
        
        # Compute terms for the triple intersection (A∩B∩C)
        termE = pow(23, n, MOD)
        termF = (n * pow(23, n-1, MOD)) % MOD
        sum_term6 = (termE + termF) % MOD
        
        # Combine all terms using inclusion-exclusion
        res = (term1 - sum_term2) % MOD
        res = (res + sum_term4) % MOD
        res = (res - sum_term6) % MOD
        
        return res % MOD