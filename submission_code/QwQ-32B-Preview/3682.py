class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        if k < 0 or k > n - 1:
            return 0
        
        # Function to compute C(n, k) modulo MOD
        def binom(n, k, MOD):
            if k < 0 or k > n:
                return 0
            numerator = 1
            for i in range(k):
                numerator = (numerator * (n - i)) % MOD
            denominator = 1
            for i in range(1, k + 1):
                denominator = (denominator * i) % MOD
            denominator_inv = pow(denominator, MOD - 2, MOD)
            return (numerator * denominator_inv) % MOD
        
        # Compute C(n-1, k)
        C = binom(n - 1, k, MOD)
        
        # Compute (m - 1)^(n - 1 - k)
        P = pow(m - 1, n - 1 - k, MOD)
        
        # Final result
        result = (C * m * P) % MOD
        return result