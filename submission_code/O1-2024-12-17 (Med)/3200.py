class Solution:
    def stringCount(self, n: int) -> int:
        """
        We count the number of length-n strings (over 26 lowercase letters) that can be
        rearranged to contain the substring "leet". Equivalently, such a string must
        have at least:
          - 1 'l'
          - 2 'e's
          - 1 't'
        
        We use Inclusion-Exclusion on the counts of strings that are missing the required
        amounts. In particular, let:
          A = set of strings with 0 'l'
          B = set of strings with < 2 'e' (i.e., 0 or 1 'e')
          C = set of strings with 0 't'
        
        We want the size of the complement of A ∪ B ∪ C, i.e. all strings - those that fail
        at least one condition. Denoting |S| = total number of length-n strings = 26^n,
        we compute:
        
          |A| = count of strings with 0 'l' = 25^n
          |B| = count of strings with 0 or 1 'e' = 25^n + n*25^(n-1)
          |C| = count of strings with 0 't' = 25^n
          
          |A ∩ B| = strings with 0 'l' and fewer than 2 'e':
                     => 0 'e': 24^n
                     => 1 'e': n*24^(n-1)
                     => total = 24^n + n*24^(n-1)
          |B ∩ C| = strings with 0 't' and fewer than 2 'e':
                     => 0 'e': 24^n
                     => 1 'e': n*24^(n-1)
                     => total = 24^n + n*24^(n-1)
          |C ∩ A| = strings with 0 't' and 0 'l' = 24^n
          
          |A ∩ B ∩ C| = strings with 0 'l', 0 't', and < 2 'e':
                        => 0 'e': 23^n
                        => 1 'e': n*23^(n-1)
                        => total = 23^n + n*23^(n-1)
        
        Then by Inclusion-Exclusion:
        
          |A ∪ B ∪ C| = |A| + |B| + |C|
                        - (|A ∩ B| + |B ∩ C| + |C ∩ A|)
                        + |A ∩ B ∩ C|
        
        Finally,
          number_of_good = 26^n - |A ∪ B ∪ C|
        
        We will compute all powers modulo (10^9 + 7) and return the result.
        
        Example checks:
          For n = 4, result should be 12 (matches the example).
          For n = 10, result should be 83943898 (matches the example).
        """

        MOD = 10**9 + 7

        # Fast power mod in Python: pow(base, exp, mod)
        pow26n = pow(26, n, MOD)
        pow25n = pow(25, n, MOD)
        pow24n = pow(24, n, MOD)
        pow23n = pow(23, n, MOD)

        # For pow(25, n-1) etc., handle n=1 carefully, though the formula is correct for n>=1
        if n > 0:
            pow25n_1 = pow(25, n-1, MOD)  # 25^(n-1)
            pow24n_1 = pow(24, n-1, MOD)  # 24^(n-1)
            pow23n_1 = pow(23, n-1, MOD)  # 23^(n-1)
        else:
            # Not actually needed since n >= 1 per the constraints
            pow25n_1 = pow24n_1 = pow23n_1 = 0

        # A = no 'l' => 25^n
        # B = fewer than 2 'e' => 25^n + n*25^(n-1)
        # C = no 't' => 25^n
        # A ∩ B => (24^n + n*24^(n-1))
        # B ∩ C => (24^n + n*24^(n-1))
        # C ∩ A => (24^n)
        # A ∩ B ∩ C => (23^n + n*23^(n-1))
        
        # Combine them using Inclusion-Exclusion:
        # good_count = 26^n
        #              - [ (3 * 25^n) + (n * 25^(n-1)) 
        #                  - (2*(24^n + n*24^(n-1)) + 24^n) 
        #                  + (23^n + n*23^(n-1)) ]
        # Carefully expanding:
        
        partA = (3 * pow25n + n * pow25n_1) % MOD              # |A| + |B| + |C|
        partB = ( (24**n) % MOD + n * (24**(n-1) % MOD ) ) % MOD  # We'll refine next lines
        # Instead of re-computing 24^n, we use the stored pow24n:
        # sum of intersections except triple:
        # (|A∩B| + |B∩C| + |C∩A|) = (24^n + n*24^(n-1)) + (24^n + n*24^(n-1)) + 24^n
        #                        = 3*24^n + 2*n*24^(n-1)
        
        # The final formula directly:
        res = (pow26n
               - 3 * pow25n
               - (n % MOD) * pow25n_1
               + 3 * pow24n
               + 2 * (n % MOD) * pow24n_1
               - pow23n
               - (n % MOD) * pow23n_1) % MOD

        return res % MOD