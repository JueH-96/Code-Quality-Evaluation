class Solution:
    def stringCount(self, n: int) -> int:
        """
        Calculates the number of "good" strings of length n using the
        Principle of Inclusion-Exclusion.

        A string is "good" if it can be rearranged to contain "leet", which means
        it must have at least one 'l', at least two 'e's, and at least one 't'.
        """
        MOD = 10**9 + 7

        # A string of length < 4 cannot contain 'l', 'e', 'e', 't'.
        if n < 4:
            return 0

        # Total number of strings of length n.
        total = pow(26, n, MOD)

        # S1: Sum of counts of strings violating one condition.
        # Violations: no 'l' (P_l), <2 'e' (P_e), no 't' (P_t).
        # |P_l| = |P_t| = 25^n
        # |P_e| = 25^n + n * 25^(n-1)
        # S1 = |P_l| + |P_t| + |P_e| = 3 * 25^n + n * 25^(n-1)
        p25_n = pow(25, n, MOD)
        p25_n_minus_1 = pow(25, n - 1, MOD)
        s1 = (3 * p25_n + n * p25_n_minus_1) % MOD

        # S2: Sum of counts of strings violating two conditions.
        # |P_l n P_t| = 24^n
        # |P_l n P_e| = |P_e n P_t| = 24^n + n * 24^(n-1)
        # S2 = 3 * 24^n + 2 * n * 24^(n-1)
        p24_n = pow(24, n, MOD)
        p24_n_minus_1 = pow(24, n - 1, MOD)
        s2 = (3 * p24_n + 2 * n * p24_n_minus_1) % MOD

        # S3: Count of strings violating all three conditions.
        # |P_l n P_e n P_t| = 23^n + n * 23^(n-1)
        p23_n = pow(23, n, MOD)
        p23_n_minus_1 = pow(23, n - 1, MOD)
        s3 = (p23_n + n * p23_n_minus_1) % MOD

        # Apply PIE: result = total - S1 + S2 - S3
        # We add MOD at each subtraction to ensure the result stays positive.
        ans = (total - s1 + MOD) % MOD
        ans = (ans + s2) % MOD
        ans = (ans - s3 + MOD) % MOD
        
        return ans