class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Edge cases
        if m == 1:
            return 1 if k == n - 1 else 0
        if k > n - 1 or k < 0:
            return 0
        
        # Calculate C(n-1, k) using modular arithmetic
        def mod_pow(base, exp, mod):
            result = 1
            base %= mod
            while exp > 0:
                if exp & 1:
                    result = (result * base) % mod
                base = (base * base) % mod
                exp >>= 1
            return result
        
        # Calculate C(n-1, k) = (n-1)! / (k! * (n-1-k)!)
        # We'll calculate this as (n-1) * (n-2) * ... * (n-k) / k!
        numerator = 1
        for i in range(n - k, n):
            numerator = (numerator * i) % MOD
        
        denominator = 1
        for i in range(1, k + 1):
            denominator = (denominator * i) % MOD
        
        # Use Fermat's little theorem for modular inverse
        inv_denominator = mod_pow(denominator, MOD - 2, MOD)
        combinations = (numerator * inv_denominator) % MOD
        
        # Calculate the final result
        power_term = mod_pow(m - 1, n - 1 - k, MOD)
        result = (combinations * m % MOD) * power_term % MOD
        
        return result