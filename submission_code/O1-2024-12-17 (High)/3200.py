class Solution:
    def stringCount(self, n: int) -> int:
        """
        We want to count the number of length-n strings over the 26 lowercase letters
        that can be rearranged to contain the substring "leet". Equivalently, such strings
        must have at least:
            - 1 'l'
            - 2 'e's
            - 1 't'
        
        Let M = 10^9 + 7.

        Total number of length-n strings (over 26 letters) = 26^n.

        We use the principle of inclusion-exclusion. Define:
          A = set of strings with 0 'l'.
          B = set of strings with at most 1 'e'.
          C = set of strings with 0 't'.

        We want to exclude A ∪ B ∪ C from 26^n.

        1) |A| = number of strings without 'l' = 25^n
        2) |B| = number of strings with at most 1 'e' = (0 'e') + (1 'e')
              = 25^n + n*25^(n-1)
        3) |C| = number of strings without 't' = 25^n

        Next, pairwise intersections:

        4) |A ∩ B|: strings with no 'l' and at most 1 'e'.
             - 0 'e': choose from 24 letters => 24^n
             - 1 'e': choose 1 position for 'e' (n ways) and the others from 24 letters => n*24^(n-1)
             => total = 24^n + n*24^(n-1)

        5) |A ∩ C|: strings with no 'l' and no 't' => choose from 24 letters => 24^n
        6) |B ∩ C|: strings with at most 1 'e' and no 't'.
             - 0 'e': choose from 24 letters => 24^n
             - 1 'e': choose 1 position for 'e' and others from 24 letters => n*24^(n-1)
             => total = 24^n + n*24^(n-1)

        Finally, triple intersection:

        7) |A ∩ B ∩ C|: strings with no 'l', at most 1 'e', and no 't'.
             - 0 'e': choose from 23 letters => 23^n
             - 1 'e': choose 1 position for 'e' and others from 23 letters => n*23^(n-1)
             => total = 23^n + n*23^(n-1)

        By inclusion-exclusion, the number of valid strings is:
            26^n
            - (|A| + |B| + |C|)
            + (|A ∩ B| + |A ∩ C| + |B ∩ C|)
            - |A ∩ B ∩ C|

        We compute all these terms modulo M and return the result.
        """

        M = 10**9 + 7
        
        # Fast exponentiations modulo M
        p26 = pow(26, n, M)
        p25 = pow(25, n, M)
        p24 = pow(24, n, M)
        p23 = pow(23, n, M)
        
        # For terms with (n-1) exponent, n >= 1
        p25_n_1 = pow(25, n-1, M)
        p24_n_1 = pow(24, n-1, M)
        p23_n_1 = pow(23, n-1, M)
        
        # |A| = 25^n
        # |B| = 25^n + n*25^(n-1)
        # |C| = 25^n
        # |A∪B∪C| = |A| + |B| + |C| - |A∩B| - |A∩C| - |B∩C| + |A∩B∩C|
        
        # Pairwise intersections:
        # |A ∩ B| = 24^n + n*24^(n-1)
        # |A ∩ C| = 24^n
        # |B ∩ C| = 24^n + n*24^(n-1)
        # Triple intersection:
        # |A ∩ B ∩ C| = 23^n + n*23^(n-1)
        
        # Apply inclusion-exclusion
        A = p25
        B = (p25 + n * p25_n_1) % M
        C = p25

        AB = (p24 + n * p24_n_1) % M
        AC = p24
        BC = (p24 + n * p24_n_1) % M
        
        ABC = (p23 + n * p23_n_1) % M
        
        # total_good = 26^n - (A + B + C) + (AB + AC + BC) - ABC
        ans = p26
        ans -= (A + B + C) % M
        ans += (AB + AC + BC) % M
        ans -= ABC
        
        ans %= M
        return ans