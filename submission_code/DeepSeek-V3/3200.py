MOD = 10**9 + 7

class Solution:
    def stringCount(self, n: int) -> int:
        # Total possible strings: 26^n
        total = pow(26, n, MOD)
        
        # Subtract strings that are missing at least one of 'l', 'e', 't'
        # Using inclusion-exclusion principle
        
        # Strings missing 'l': 25^n
        no_l = pow(25, n, MOD)
        
        # Strings missing 'e': 25^n
        no_e = pow(25, n, MOD)
        
        # Strings missing 't': 25^n
        no_t = pow(25, n, MOD)
        
        # Strings missing both 'l' and 'e': 24^n
        no_l_e = pow(24, n, MOD)
        
        # Strings missing both 'l' and 't': 24^n
        no_l_t = pow(24, n, MOD)
        
        # Strings missing both 'e' and 't': 24^n
        no_e_t = pow(24, n, MOD)
        
        # Strings missing all 'l', 'e', 't': 23^n
        no_l_e_t = pow(23, n, MOD)
        
        # Applying inclusion-exclusion
        # total - (no_l + no_e + no_t) + (no_l_e + no_l_t + no_e_t) - no_l_e_t
        result = (total - (no_l + no_e + no_t) + (no_l_e + no_l_t + no_e_t) - no_l_e_t) % MOD
        
        # Now, we need to ensure that the string has at least two 'e's
        # Because 'leet' requires at least two 'e's
        
        # Total strings with at least two 'e's: total - (strings with 0 'e's + strings with 1 'e')
        # strings with 0 'e's: 25^n
        # strings with 1 'e': n * 25^(n-1)
        no_e = pow(25, n, MOD)
        one_e = n * pow(25, n-1, MOD) % MOD
        at_least_two_e = (total - no_e - one_e) % MOD
        
        # The final result is the intersection of the two conditions:
        # 1. Contains at least one 'l', one 'e', one 't'
        # 2. Contains at least two 'e's
        # So, the result is the number of strings that satisfy both conditions
        
        # To find the intersection, we can use the inclusion-exclusion principle again
        # Let A be the set of strings that contain at least one 'l', one 'e', one 't'
        # Let B be the set of strings that contain at least two 'e's
        # We need |A ∩ B| = |A| + |B| - |A ∪ B|
        
        # |A| is the result calculated above
        # |B| is at_least_two_e
        # |A ∪ B| is the total number of strings that either contain at least one 'l', one 'e', one 't' or contain at least two 'e's
        
        # To find |A ∪ B|, we can use the inclusion-exclusion principle:
        # |A ∪ B| = |A| + |B| - |A ∩ B|
        # But we need to find |A ∩ B|, so we need to rearrange:
        # |A ∩ B| = |A| + |B| - |A ∪ B|
        
        # Alternatively, we can think of |A ∩ B| as the number of strings that contain at least one 'l', one 't', and at least two 'e's
        
        # So, we can calculate |A ∩ B| directly:
        # Total strings: 26^n
        # Subtract strings missing 'l': 25^n
        # Subtract strings missing 't': 25^n
        # Subtract strings with less than two 'e's: 25^n + n * 25^(n-1)
        # Add back strings missing both 'l' and 't': 24^n
        # Add back strings missing 'l' and have less than two 'e's: 24^n + n * 24^(n-1)
        # Add back strings missing 't' and have less than two 'e's: 24^n + n * 24^(n-1)
        # Subtract strings missing both 'l' and 't' and have less than two 'e's: 23^n + n * 23^(n-1)
        
        # So, the formula is:
        # |A ∩ B| = 26^n - 25^n - 25^n - (25^n + n * 25^(n-1)) + 24^n + (24^n + n * 24^(n-1)) + (24^n + n * 24^(n-1)) - (23^n + n * 23^(n-1))
        
        # Simplifying:
        # |A ∩ B| = 26^n - 3 * 25^n - n * 25^(n-1) + 3 * 24^n + 2 * n * 24^(n-1) - 23^n - n * 23^(n-1)
        
        # Calculating each term:
        term1 = pow(26, n, MOD)
        term2 = 3 * pow(25, n, MOD) % MOD
        term3 = n * pow(25, n-1, MOD) % MOD
        term4 = 3 * pow(24, n, MOD) % MOD
        term5 = 2 * n * pow(24, n-1, MOD) % MOD
        term6 = pow(23, n, MOD)
        term7 = n * pow(23, n-1, MOD) % MOD
        
        intersection = (term1 - term2 - term3 + term4 + term5 - term6 - term7) % MOD
        
        return intersection