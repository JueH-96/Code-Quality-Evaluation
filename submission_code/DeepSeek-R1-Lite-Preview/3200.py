class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        if n < 4:
            return 0
        
        # Compute powers using fast exponentiation
        def power(a, b):
            return pow(a, b, MOD)
        
        term1 = power(26, n)
        term2 = (3 * power(25, n)) % MOD
        term3 = (n * power(25, n-1)) % MOD
        term4 = (2 * power(24, n)) % MOD
        term5 = (n * power(24, n-1)) % MOD
        
        # Compute the final result using inclusion-exclusion formula
        result = (term1 - term2 - term3 + term4 + term5) % MOD
        return result