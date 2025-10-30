class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7

        # n_mod is used for the coefficient 'n' to ensure it's within MOD range for arithmetic operations
        n_mod = n % MOD

        # Calculate powers for the bases 26, 25, 24, 23, representing number of character choices
        p26_n = pow(26, n, MOD)
        p25_n = pow(25, n, MOD)
        p24_n = pow(24, n, MOD)
        p23_n = pow(23, n, MOD)

        # Calculate powers for n-1 for cases where 0 or 1 specific character is allowed
        # These are used in the terms involving (n * base^(n-1))
        p25_n_minus_1 = pow(25, n - 1, MOD)
        p24_n_minus_1 = pow(24, n - 1, MOD)
        p23_n_minus_1 = pow(23, n - 1, MOD)

        # Inclusion-Exclusion Principle:
        # We want to find the number of strings that satisfy:
        # (count of 'l' >= 1) AND (count of 'e' >= 2) AND (count of 't' >= 1)
        # Let P_l be the property that count('l') == 0
        # Let P_e be the property that count('e') < 2 (i.e., 0 or 1)
        # Let P_t be the property that count('t') == 0
        # We are looking for N(U) - |P_l U P_e U P_t|
        # By Inclusion-Exclusion:
        # |P_l U P_e U P_t| = Sum(|P_i|) - Sum(|P_i intersect P_j|) + |P_l intersect P_e intersect P_t|

        # S1: Sum of counts of strings with one property (P_l, P_e, P_t)
        # N(P_l) = 25^n
        # N(P_t) = 25^n
        # N(P_e) = 25^n + n * 25^(n-1) (for 0 'e's or 1 'e')
        # S1 = N(P_l) + N(P_t) + N(P_e)
        S1 = (p25_n + p25_n + (p25_n + (n_mod * p25_n_minus_1) % MOD)) % MOD

        # S2: Sum of counts of strings with two properties (intersections)
        # N(P_l AND P_t) = 24^n
        # N(P_l AND P_e) = 24^n + n * 24^(n-1)
        # N(P_t AND P_e) = 24^n + n * 24^(n-1)
        # S2 = N(P_l AND P_t) + N(P_l AND P_e) + N(P_t AND P_e)
        S2 = (p24_n + (p24_n + (n_mod * p24_n_minus_1) % MOD) + (p24_n + (n_mod * p24_n_minus_1) % MOD)) % MOD

        # S3: Count of strings with all three properties (intersection)
        # N(P_l AND P_e AND P_t) = 23^n + n * 23^(n-1)
        S3 = (p23_n + (n_mod * p23_n_minus_1) % MOD) % MOD

        # Final calculation using Inclusion-Exclusion Principle
        # Result = Total strings (p26_n) - S1 + S2 - S3
        # Ensure results stay positive by adding MOD before taking modulo for subtractions
        ans = (p26_n - S1 + MOD) % MOD
        ans = (ans + S2) % MOD
        ans = (ans - S3 + MOD) % MOD
        
        return ans