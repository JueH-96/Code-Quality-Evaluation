MOD = 10**9 + 7

class Solution:
    def stringCount(self, n: int) -> int:
        # Total possible strings: 26^n
        total = pow(26, n, MOD)
        
        # Subtract strings that are missing at least one of 'l', 'e', 't'
        # Using inclusion-exclusion principle
        
        # Strings missing 'l': 25^n
        missing_l = pow(25, n, MOD)
        
        # Strings missing 'e': 25^n
        missing_e = pow(25, n, MOD)
        
        # Strings missing 't': 25^n
        missing_t = pow(25, n, MOD)
        
        # Strings missing both 'l' and 'e': 24^n
        missing_l_e = pow(24, n, MOD)
        
        # Strings missing both 'l' and 't': 24^n
        missing_l_t = pow(24, n, MOD)
        
        # Strings missing both 'e' and 't': 24^n
        missing_e_t = pow(24, n, MOD)
        
        # Strings missing all 'l', 'e', 't': 23^n
        missing_l_e_t = pow(23, n, MOD)
        
        # Applying inclusion-exclusion
        # total - (missing_l + missing_e + missing_t) + (missing_l_e + missing_l_t + missing_e_t) - missing_l_e_t
        result = (total - (missing_l + missing_e + missing_t) + (missing_l_e + missing_l_t + missing_e_t) - missing_l_e_t) % MOD
        
        return result