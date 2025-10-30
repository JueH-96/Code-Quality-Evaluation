import math

class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7

        # The maximum value for n_val in nCr(n_val, r_val) will be n-1.
        # So we need factorials up to n-1.
        # k can be up to n-1.
        max_val_for_factorial = n - 1 
        
        # Initialize factorials and inverse factorials arrays.
        # fact[0]=1, invFact[0]=1 are base cases for factorials/inverse factorials.
        # The arrays are sized to accommodate max_val_for_factorial (up to 10^5 - 1).
        fact = [1] * (max_val_for_factorial + 1)
        invFact = [1] * (max_val_for_factorial + 1)

        # Modular exponentiation function: (base^exp) % MOD
        # Handles 0^0 = 1 and 0^positive = 0 correctly.
        def power(base, exp):
            res = 1
            base %= MOD # Ensure base is within [0, MOD-1]
            
            # Special handling for 0^0 and 0^positive cases
            if base == 0 and exp == 0:
                return 1 # Mathematical convention for 0^0
            if base == 0:
                return 0 # 0 raised to any positive integer is 0
            
            # Standard modular exponentiation
            while exp > 0:
                if exp % 2 == 1:
                    res = (res * base) % MOD
                base = (base * base) % MOD
                exp //= 2
            return res

        # Precompute factorials: fact[i] = i! % MOD
        # Loop runs from 1 up to max_val_for_factorial
        for i in range(1, max_val_for_factorial + 1):
            fact[i] = (fact[i-1] * i) % MOD
        
        # Precompute inverse factorials: invFact[i] = (i!)^(-1) % MOD
        # Using Fermat's Little Theorem: a^(p-2) % p is modular multiplicative inverse of a for prime p
        # Compute invFact for the largest factorial first, then iterate downwards.
        # Ensure max_val_for_factorial is non-negative before accessing fact array.
        if max_val_for_factorial >= 0:
            invFact[max_val_for_factorial] = power(fact[max_val_for_factorial], MOD - 2)
            # Iterate downwards to compute other inverse factorials: invFact[i] = invFact[i+1] * (i+1) % MOD
            for i in range(max_val_for_factorial - 1, -1, -1):
                invFact[i] = (invFact[i+1] * (i+1)) % MOD

        # Function to calculate nCr % MOD using precomputed factorials and inverse factorials
        def nCr_mod_p(n_val, r_val):
            # If r is out of valid range, combinations is 0
            if r_val < 0 or r_val > n_val:
                return 0
            
            # C(n_val, r_val) = n_val! / (r_val! * (n_val - r_val)!)
            # In modular arithmetic: (n_val! * (r_val!)^(-1) * ((n_val - r_val)!)^(-1)) % MOD
            numerator = fact[n_val]
            denominator = (invFact[r_val] * invFact[n_val - r_val]) % MOD
            return (numerator * denominator) % MOD

        # The derived formula is: C(n-1, k) * m * (m-1)^(n-1-k)
        
        # Part 1: Calculate C(n-1, k)
        combinations = nCr_mod_p(n - 1, k)

        # Part 2: The term 'm'
        term_m = m % MOD

        # Part 3: Calculate (m-1)^(n-1-k)
        exponent = n - 1 - k
        base_for_power = (m - 1) % MOD
        
        term_m_minus_1_power = power(base_for_power, exponent)

        # Combine all parts, taking modulo at each multiplication step to prevent overflow
        ans = (combinations * term_m) % MOD
        ans = (ans * term_m_minus_1_power) % MOD

        return ans