class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7

        def power(base, exp, mod):
            res = 1
            base %= mod
            while exp > 0:
                if exp % 2 == 1:
                    res = (res * base) % mod
                base = (base * base) % mod
                exp //= 2
            return res

        # Total number of strings of length n using 26 lowercase English characters is 26^n.
        total_count = power(26, n, MOD)

        # A string is "good" if it can be rearranged to contain "leet" as a substring.
        # This is possible if and only if the string contains at least one 'l',
        # at least two 'e's, and at least one 't'.
        # Let Cl, Ce, Ct be the counts of 'l', 'e', 't' respectively.
        # A string is good if Cl >= 1 AND Ce >= 2 AND Ct >= 1.

        # It's easier to count the complement: strings that are NOT good.
        # A string is not good if Cl < 1 OR Ce < 2 OR Ct < 1.
        # This is equivalent to Cl = 0 OR Ce <= 1 OR Ct = 0.

        # Let the properties for "not good" strings be:
        # P1: Cl = 0
        # P2: Ce <= 1 (which means Ce=0 or Ce=1)
        # P3: Ct = 0

        # We want to find the number of strings satisfying P1 OR P2 OR P3.
        # Using the Principle of Inclusion-Exclusion (PIE):
        # |P1 U P2 U P3| = |P1| + |P2| + |P3| - (|P1 n P2| + |P1 n P3| + |P2 n P3|) + |P1 n P2 n P3|

        # The number of good strings is Total - |P1 U P2 U P3|.

        n_mod = n % MOD

        # |P1|: Strings with Cl = 0. Each of the n positions can be filled with any character
        # except 'l'. There are 25 such characters. Number of strings = 25^n.
        count_P1 = power(25, n, MOD)

        # |P3|: Strings with Ct = 0. Each of the n positions can be filled with any character
        # except 't'. There are 25 such characters. Number of strings = 25^n.
        count_P3 = power(25, n, MOD)

        # |P2|: Strings with Ce <= 1. This is the sum of strings with Ce=0 and Ce=1.
        # Strings with Ce = 0: Use any of 25 characters (not 'e'). 25^n ways.
        # Strings with Ce = 1: Choose 1 position for 'e' (n ways). The remaining n-1 positions
        # can be filled with any of the other 25 characters (not 'e'). n * 25^(n-1) ways.
        # Handle n-1 = 0 case: power(base, 0, mod) = 1. n * power(base, 0, mod) = n.
        count_P2 = (power(25, n, MOD) + n_mod * power(25, n-1, MOD)) % MOD

        # Sum of individuals: |P1| + |P2| + |P3|
        sum_individuals = (count_P1 + count_P2 + count_P3) % MOD

        # |P1 n P2|: Strings with Cl = 0 AND Ce <= 1.
        # These strings use characters from the 24 letters (not 'l', 'e').
        # Cl=0, Ce=0: Use 24 chars. 24^n ways.
        # Cl=0, Ce=1: Choose 1 pos for 'e' (n ways). Fill rest with 24 chars. n * 24^(n-1) ways.
        count_P1nP2 = (power(24, n, MOD) + n_mod * power(24, n-1, MOD)) % MOD

        # |P1 n P3|: Strings with Cl = 0 AND Ct = 0.
        # These strings use characters from the 24 letters (not 'l', 't'). 24^n ways.
        count_P1nP3 = power(24, n, MOD)

        # |P2 n P3|: Strings with Ce <= 1 AND Ct = 0.
        # These strings use characters from the 24 letters (not 'e', 't').
        # Ce=0, Ct=0: Use 24 chars. 24^n ways.
        # Ce=1, Ct=0: Choose 1 pos for 'e' (n ways). Fill rest with 24 chars. n * 24^(n-1) ways.
        count_P2nP3 = (power(24, n, MOD) + n_mod * power(24, n-1, MOD)) % MOD

        # Sum of pairwise intersections: |P1 n P2| + |P1 n P3| + |P2 n P3|
        sum_pairwise = (count_P1nP2 + count_P1nP3 + count_P2nP3) % MOD

        # |P1 n P2 n P3|: Strings with Cl = 0 AND Ce <= 1 AND Ct = 0.
        # These strings use characters from the 23 letters (not 'l', 'e', 't').
        # Cl=0, Ce=0, Ct=0: Use 23 chars. 23^n ways.
        # Cl=0, Ce=1, Ct=0: Choose 1 pos for 'e' (n ways). Fill rest with 23 chars. n * 23^(n-1) ways.
        count_P1nP2nP3 = (power(23, n, MOD) + n_mod * power(23, n-1, MOD)) % MOD

        # Number of not good strings = |P1 U P2 U P3| = Sum_individuals - Sum_pairwise + count_P1nP2nP3
        # Perform subtraction with modulo carefully to handle negative intermediate results.
        not_good_count = (sum_individuals - sum_pairwise + MOD) % MOD
        not_good_count = (not_good_count + count_P1nP2nP3) % MOD

        # Number of good strings = Total strings - Number of not good strings
        good_count = (total_count - not_good_count + MOD) % MOD

        return good_count