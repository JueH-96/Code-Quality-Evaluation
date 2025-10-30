class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        # Compute powers
        p26 = pow(26, n, MOD)
        p25 = pow(25, n, MOD)
        p24 = pow(24, n, MOD)
        p23 = pow(23, n, MOD)
        
        # Powers for the n-1 exponents
        # (for the terms counting exactly one 'e')
        p25_1 = pow(25, n-1, MOD)
        p24_1 = pow(24, n-1, MOD)
        p23_1 = pow(23, n-1, MOD)
        
        # Inclusion-exclusion breakdown:
        # invalid = 3*25^n - 3*24^n + 23^n
        #         + n * (25^(n-1) - 2*24^(n-1) + 23^(n-1))
        const_part = (3*p25 - 3*p24 + p23) % MOD
        linear_part = n * ((p25_1 - 2*p24_1 + p23_1) % MOD) % MOD
        invalid = (const_part + linear_part) % MOD
        
        # Total valid = 26^n - invalid
        return (p26 - invalid) % MOD