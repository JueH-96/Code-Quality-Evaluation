class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        # Helper function to compute power modulo MOD
        def pow_mod(base, exponent):
            return pow(base, exponent, MOD)
        
        # Total number of strings
        total = pow_mod(26, n)
        
        # Number of strings with fewer than 2 'l's
        A = pow_mod(25, n) + n * pow_mod(25, n - 1) if n >= 1 else 0
        # Number of strings with fewer than 2 'e's
        B = pow_mod(25, n) + n * pow_mod(25, n - 1) if n >= 1 else 0
        # Number of strings with fewer than 1 't'
        C = pow_mod(25, n)
        
        # Number of strings with fewer than 2 'l's and fewer than 2 'e's
        if n >= 2:
            A_inter_B = (pow_mod(24, n) +
                         2 * n * pow_mod(24, n - 1) +
                         n * (n - 1) // 2 * pow_mod(24, n - 2))
        elif n == 1:
            A_inter_B = pow_mod(24, n) + 2 * n * pow_mod(24, n - 1)
        else:
            A_inter_B = 0
        
        # Number of strings with fewer than 2 'l's and fewer than 1 't'
        if n >= 1:
            A_inter_C = pow_mod(24, n) + n * pow_mod(23, n - 1)
            B_inter_C = pow_mod(24, n) + n * pow_mod(23, n - 1)
        else:
            A_inter_C = 0
            B_inter_C = 0
        
        # Number of strings with fewer than 2 'l's, fewer than 2 'e's, and fewer than 1 't'
        if n >= 2:
            A_inter_B_inter_C = (pow_mod(23, n) +
                                 2 * n * pow_mod(23, n - 1) +
                                 n * (n - 1) // 2 * pow_mod(23, n - 2))
        elif n == 1:
            A_inter_B_inter_C = pow_mod(23, n) + 2 * n * pow_mod(23, n - 1)
        else:
            A_inter_B_inter_C = 0
        
        # Inclusion-Exclusion
        bad = (A + B + C - A_inter_B - A_inter_C - B_inter_C + A_inter_B_inter_C) % MOD
        
        # Good strings
        good = (total - bad) % MOD
        
        return good