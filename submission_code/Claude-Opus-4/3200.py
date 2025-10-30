class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        # Total strings of length n
        total = pow(26, n, MOD)
        
        # Strings with no 'l'
        no_l = pow(25, n, MOD)
        
        # Strings with 0 or 1 'e'
        # 0 'e's: 25^n
        # 1 'e': n * 25^(n-1)
        at_most_one_e = (pow(25, n, MOD) + n * pow(25, n-1, MOD)) % MOD
        
        # Strings with no 't'
        no_t = pow(25, n, MOD)
        
        # Strings with no 'l' AND (0 or 1 'e')
        # 0 'e's: 24^n
        # 1 'e': n * 24^(n-1)
        no_l_and_at_most_one_e = (pow(24, n, MOD) + n * pow(24, n-1, MOD)) % MOD
        
        # Strings with no 'l' AND no 't'
        no_l_and_no_t = pow(24, n, MOD)
        
        # Strings with (0 or 1 'e') AND no 't'
        # 0 'e's: 24^n
        # 1 'e': n * 24^(n-1)
        at_most_one_e_and_no_t = (pow(24, n, MOD) + n * pow(24, n-1, MOD)) % MOD
        
        # Strings with no 'l' AND (0 or 1 'e') AND no 't'
        # 0 'e's: 23^n
        # 1 'e': n * 23^(n-1)
        no_l_and_at_most_one_e_and_no_t = (pow(23, n, MOD) + n * pow(23, n-1, MOD)) % MOD
        
        # Apply inclusion-exclusion
        bad = (no_l + at_most_one_e + no_t 
               - no_l_and_at_most_one_e - no_l_and_no_t - at_most_one_e_and_no_t
               + no_l_and_at_most_one_e_and_no_t) % MOD
        
        result = (total - bad) % MOD
        
        # Ensure non-negative result
        if result < 0:
            result += MOD
            
        return result