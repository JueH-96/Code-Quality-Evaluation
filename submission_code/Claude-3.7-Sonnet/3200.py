class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        # Total number of strings of length n
        total = pow(26, n, MOD)
        
        # Count strings missing required characters:
        # A: strings without 'l'
        A = pow(25, n, MOD)
        
        # B: strings with less than 2 'e's
        B_0e = pow(25, n, MOD)  # Strings with 0 'e's
        B_1e = (n * pow(25, n-1, MOD)) % MOD  # Strings with exactly 1 'e'
        B = (B_0e + B_1e) % MOD
        
        # C: strings without 't'
        C = pow(25, n, MOD)
        
        # Calculate intersections using inclusion-exclusion principle
        # A ∩ B: strings without 'l' and with less than 2 'e's
        AB_0e = pow(24, n, MOD)  # Strings without 'l' and with 0 'e's
        AB_1e = (n * pow(24, n-1, MOD)) % MOD  # Strings without 'l' and with 1 'e'
        A_B = (AB_0e + AB_1e) % MOD
        
        # A ∩ C: strings without 'l' and 't'
        A_C = pow(24, n, MOD)
        
        # B ∩ C: strings with less than 2 'e's and without 't'
        BC_0e = pow(24, n, MOD)  # Strings with 0 'e's and without 't'
        BC_1e = (n * pow(24, n-1, MOD)) % MOD  # Strings with 1 'e' and without 't'
        B_C = (BC_0e + BC_1e) % MOD
        
        # A ∩ B ∩ C: strings without 'l', with less than 2 'e's, and without 't'
        ABC_0e = pow(23, n, MOD)  # Strings without 'l', 't', and with 0 'e's
        ABC_1e = (n * pow(23, n-1, MOD)) % MOD  # Strings without 'l', 't', and with 1 'e'
        A_B_C = (ABC_0e + ABC_1e) % MOD
        
        # Using inclusion-exclusion principle to find bad strings
        term1 = (A + B) % MOD
        term1 = (term1 + C) % MOD
        term2 = (A_B + A_C) % MOD
        term2 = (term2 + B_C) % MOD
        bad_strings = (term1 - term2 + A_B_C) % MOD
        
        # Number of good strings = total - bad_strings
        good_strings = (total - bad_strings) % MOD
            
        return good_strings