class Solution:
    def stringCount(self, n: int) -> int:
        """
        We say a string s (consisting of only lowercase English letters) is "good" 
        if we can rearrange its characters so that the substring "leet" appears.

        Equivalently, s must contain at least:
          • 1 'l'
          • 2 'e's
          • 1 't'
        anywhere in the string (not necessarily contiguous), because if it does,
        we can always rearrange so that "leet" is a contiguous substring.

        We want the number of such strings of length n, modulo 10^9+7.
        
        --------------------------------------------------------------------------
        Inclusion-Exclusion Approach:

        Let:
          • S = set of all length-n strings over 26 lowercase letters  ⇒ |S| = 26^n
          • A = set of strings with no 'l'
          • B = set of strings with fewer than 2 'e's (i.e., 0 or 1 'e')
          • C = set of strings with no 't'

        We want to count the complement of (A ∪ B ∪ C), i.e. all strings that have
        at least 1 'l', at least 2 'e's, and at least 1 't'.

        By Inclusion-Exclusion:
          |A ∪ B ∪ C| = |A| + |B| + |C|
                         - (|A ∩ B| + |B ∩ C| + |A ∩ C|)
                         + |A ∩ B ∩ C|
        
        Then the answer = |S| - |A ∪ B ∪ C|.

        Let's break down each count:

        • |A|: strings with no 'l' 
            => We can only use 25 letters ⇒ 25^n

        • |C|: strings with no 't'
            => Similarly 25^n

        • |B|: strings with fewer than 2 'e's 
            => e=0 or e=1:
               e=0: choose from 25 letters (everything but 'e') ⇒ 25^n
               e=1: choose 1 position out of n for 'e', and the other (n-1) positions 
                     from 25 letters ⇒ n * 25^(n-1)
               So, |B| = 25^n + n * 25^(n-1)

        Intersections:
        • |A ∩ C|: no 'l' and no 't' 
            => use 24 remaining letters ⇒ 24^n

        • |A ∩ B|: no 'l' and fewer than 2 'e's
            - no 'l' leaves us 25 letters, among which 'e' is still included. Then e=0 or e=1:
              e=0: we cannot use 'l' or 'e', so 24 letters ⇒ 24^n
              e=1: choose 1 position for 'e' (n ways), other positions from 24 letters 
                    ⇒ n * 24^(n-1)
              So, |A ∩ B| = 24^n + n * 24^(n-1)

        • |B ∩ C|: fewer than 2 'e's and no 't'
            => analogous to above, we end up with 24^n + n * 24^(n-1)

        • |A ∩ B ∩ C|: no 'l', no 't', and fewer than 2 'e's
            => among the 26 letters, remove 'l' and 't' ⇒ 24 letters remain, among which 'e' is still included.
               e=0: then we cannot use 'e' either ⇒ 23 letters ⇒ 23^n
               e=1: choose 1 position for 'e', the rest from 23 letters ⇒ n * 23^(n-1)
               So, |A ∩ B ∩ C| = 23^n + n * 23^(n-1)

        Putting it together:

          let mod = 10^9 + 7

          total = 26^n
          bad   = |A| + |B| + |C| 
                   - (|A ∩ B| + |B ∩ C| + |A ∩ C|) 
                   + |A ∩ B ∩ C|

          good  = total - bad

          i.e.

          good = 26^n
                  - [ 25^n + (25^n + n*25^(n-1)) + 25^n ]
                  + [ (24^n + n*24^(n-1)) + (24^n + n*24^(n-1)) + 24^n ]
                  - [ 23^n + n*23^(n-1) ]

               = 26^n
                 - [ 3*25^n + n*25^(n-1) ]
                 + [ 3*24^n + 2*n*24^(n-1) ]
                 - [ 23^n + n*23^(n-1) ]

        We'll compute this expression using modular exponentiation.

        Example check: n=4 gives 12, matching the problem statement.
        Example check: n=10 => 83943898 (as provided).
        """

        MOD = 10**9 + 7

        # Fast exponentiation in Python can be done by built-in pow(base, exp, mod)
        p26n = pow(26, n, MOD)
        p25n = pow(25, n, MOD)
        p25n_1 = pow(25, n - 1, MOD) if n > 0 else 0  # for n=1, 25^(0)=1
        p24n = pow(24, n, MOD)
        p24n_1 = pow(24, n - 1, MOD) if n > 0 else 0
        p23n = pow(23, n, MOD)
        p23n_1 = pow(23, n - 1, MOD) if n > 0 else 0

        # Apply the formula:
        # good = 26^n
        #        - [3*25^n + n*25^(n-1)]
        #        + [3*24^n + 2*n*24^(n-1)]
        #        - [23^n + n*23^(n-1)]
        ans = p26n
        ans -= (3 * p25n + n * p25n_1) % MOD
        ans += (3 * p24n + 2 * n * p24n_1) % MOD
        ans -= (p23n + n * p23n_1) % MOD

        ans %= MOD  # ensure non-negative modulo

        return ans