class Solution:
    def stringCount(self, n: int) -> int:
        """
        We want the number of length-n strings (over the 26 lowercase letters) 
        which can be rearranged to contain the substring "leet". 
        Equivalently, such strings must contain at least:
            - 1 'l'
            - 2 'e's
            - 1 't'
        
        By inclusion-exclusion, we count all strings minus those that violate 
        one or more of the requirements.
        
        Let:
          A = set of strings with no 'l'
          B = set of strings with no 't'
          C = set of strings with at most 1 'e'
        
        Then the count of valid strings is:
        
            Total = 26^n
                   - (|A| + |B| + |C|)
                   + (|A ∩ B| + |A ∩ C| + |B ∩ C|)
                   - |A ∩ B ∩ C|
        
        Where:
          |A| = 25^n  (exclude 'l')
          |B| = 25^n  (exclude 't')
          |C| = number of strings with at most 1 'e':
                  = 25^n   (exclude 'e' entirely)
                    + n * 25^(n-1)  (exactly 1 'e' in one of n positions)
        
          |A ∩ B| = 24^n   (exclude 'l' and 't')
          |A ∩ C| = number of strings with no 'l' and at most 1 'e':
                     = 24^n              (exclude 'l' and 'e')
                       + n * 24^(n-1)    (exactly 1 'e', exclude 'l' from others)
          |B ∩ C| = number of strings with no 't' and at most 1 'e':
                     = 24^n
                       + n * 24^(n-1)
        
          |A ∩ B ∩ C| = number of strings with no 'l', no 't', and at most 1 'e':
                          = 23^n               (exclude 'l','t','e')
                            + n * 23^(n-1)
        
        Putting it all together yields the formula:

            count = 26^n
                    - 3 * 25^n
                    - n * 25^(n-1)
                    + 3 * 24^n
                    + 2 * n * 24^(n-1)
                    - 23^n
                    - n * 23^(n-1)

        and we take this result modulo (10^9 + 7).
        """
        mod = 10**9 + 7

        # Fast exponentiation modulo for bases 23, 24, 25, 26
        p26n = pow(26, n, mod)
        p25n = pow(25, n, mod)
        p24n = pow(24, n, mod)
        p23n = pow(23, n, mod)

        # For the terms like 25^(n-1), handle the case n>=1:
        if n >= 1:
            p25n_1 = pow(25, n-1, mod)
            p24n_1 = pow(24, n-1, mod)
            p23n_1 = pow(23, n-1, mod)
        else:
            # Not really needed for constraints n>=1
            p25n_1 = p24n_1 = p23n_1 = 0

        # Apply the inclusion-exclusion formula
        ans = (p26n
               - 3 * p25n
               - n * p25n_1
               + 3 * p24n
               + 2 * n * p24n_1
               - p23n
               - n * p23n_1)

        ans %= mod
        return ans