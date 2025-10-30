class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        def pow_mod(base, exp, mod):
            result = 1
            base %= mod
            while exp > 0:
                if exp & 1:
                    result = (result * base) % mod
                exp >>= 1
                base = (base * base) % mod
            return result
        
        # Total possible strings
        total = pow_mod(26, n, MOD)
        
        # Calculate powers we need
        pow25_n = pow_mod(25, n, MOD)
        pow25_n1 = pow_mod(25, n-1, MOD) if n > 0 else 0
        pow24_n = pow_mod(24, n, MOD)
        pow24_n1 = pow_mod(24, n-1, MOD) if n > 0 else 0
        pow23_n = pow_mod(23, n, MOD)
        pow23_n1 = pow_mod(23, n-1, MOD) if n > 0 else 0
        
        # |A| = strings with no 'l'
        A = pow25_n
        
        # |B| = strings with 0 or 1 'e'
        B = (pow25_n + (n * pow25_n1) % MOD) % MOD
        
        # |C| = strings with no 't'
        C = pow25_n
        
        # |A ∩ B| = strings with no 'l' and (0 or 1 'e')
        AB = (pow24_n + (n * pow24_n1) % MOD) % MOD
        
        # |A ∩ C| = strings with no 'l' and no 't'
        AC = pow24_n
        
        # |B ∩ C| = strings with no 't' and (0 or 1 'e')
        BC = (pow24_n + (n * pow24_n1) % MOD) % MOD
        
        # |A ∩ B ∩ C| = strings with no 'l', no 't', and (0 or 1 'e')
        ABC = (pow23_n + (n * pow23_n1) % MOD) % MOD
        
        # Inclusion-exclusion
        bad = (A + B + C - AB - AC - BC + ABC) % MOD
        
        return (total - bad + MOD) % MOD