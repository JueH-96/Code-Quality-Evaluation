class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        # If n < 4, it's impossible to have at least 1 'l', 2 'e's, and 1 't'
        if n < 4:
            return 0
        
        # Fast modular exponentiation
        pow26 = pow(26, n, MOD)
        pow25 = pow(25, n, MOD)
        pow24 = pow(24, n, MOD)
        pow23 = pow(23, n, MOD)
        
        # Precompute the n*base^(n-1) terms
        # Careful when n=0,1, but here n>=4 so safe.
        term25_n_1 = n * pow(25, n-1, MOD) % MOD
        term24_n_1 = n * pow(24, n-1, MOD) % MOD
        term23_n_1 = n * pow(23, n-1, MOD) % MOD
        
        # |A|: no 'l'
        A = pow25
        # |C|: no 't'
        C = pow25
        # |B|: at most one 'e' => 0 'e's or 1 'e'
        B = (pow25 + term25_n_1) % MOD
        
        # A ∩ C: no 'l', no 't' => 24^n
        AC = pow24
        # A ∩ B: no 'l' and at most one 'e'
        AB = (pow24 + term24_n_1) % MOD
        # B ∩ C: no 't' and at most one 'e'
        BC = AB  # same count as AB
        
        # A ∩ B ∩ C: no 'l', no 't', at most one 'e'
        ABC = (pow23 + term23_n_1) % MOD
        
        # Inclusion-Exclusion to count bad strings
        bad = (A + B + C) % MOD
        bad = (bad - AB - AC - BC) % MOD
        bad = (bad + ABC) % MOD
        
        # Good = total - bad
        ans = (pow26 - bad) % MOD
        return ans