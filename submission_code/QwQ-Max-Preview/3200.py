class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        if n < 4:
            return 0
        
        # Calculate each term using modular exponentiation
        term1 = pow(26, n, MOD)
        
        term2 = (3 * pow(25, n, MOD)) % MOD
        term3 = (n * pow(25, n - 1, MOD)) % MOD
        part2 = (term2 + term3) % MOD
        result = (term1 - part2) % MOD
        
        term4 = (3 * pow(24, n, MOD)) % MOD
        term5 = (2 * n * pow(24, n - 1, MOD)) % MOD
        part3 = (term4 + term5) % MOD
        result = (result + part3) % MOD
        
        term6 = pow(23, n, MOD)
        term7 = (n * pow(23, n - 1, MOD)) % MOD
        part4 = (term6 + term7) % MOD
        result = (result - part4) % MOD
        
        # Ensure the result is non-negative
        return result % MOD