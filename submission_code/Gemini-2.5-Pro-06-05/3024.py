class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        """
        Calculates the number of ways to transform s to t in k steps.

        An operation is a cyclic shift by l positions, where l is in {1, ..., n-1}.
        The method relies on a dynamic programming approach on the states, where
        states are the distinct cyclic shifts of s. The transitions between states
        are uniform due to symmetry. This leads to a system of linear recurrences
        which can be solved in closed form.

        Let p_s be the number of distinct cyclic shifts of s.
        The closed-form solutions are:
        If target is s: ways = ( (n-1)^k + (p_s-1)*(-1)^k ) / p_s
        If target is t!=s: ways = ( (n-1)^k - (-1)^k ) / p_s
        """
        MOD = 10**9 + 7
        n = len(s)

        # Step 1: Check if t is a cyclic shift of s.
        # If not, it's impossible to transform s to t.
        # s + s contains all cyclic shifts of s.
        if t not in s + s:
            return 0

        # Step 2: Find the number of distinct cyclic shifts of s.
        # This is equal to the length of the smallest period of s,
        # which can be found using the KMP prefix function.
        pi = self._compute_pi(s)
        lps = pi[n - 1]  # length of longest proper prefix that is also a suffix
        p_s = 0
        if n % (n - lps) == 0:
            p_s = n - lps
        else:
            p_s = n

        # Step 3: Calculate the number of ways using the derived formulas.
        # We need to compute (a / b) mod p, which is (a * b^(p-2)) mod p.
        
        # (n-1)^k mod MOD
        term1 = pow(n - 1, k, MOD)
        # (-1)^k mod MOD
        term2_sign = 1 if k % 2 == 0 else MOD - 1

        # Modular inverse of p_s
        p_s_inv = pow(p_s, MOD - 2, MOD)

        if s == t:
            # Formula for s == t: ways = ( (n-1)^k + (p_s-1)*(-1)^k ) / p_s
            # (p_s - 1) * (-1)^k mod MOD
            term2 = ((p_s - 1) % MOD * term2_sign) % MOD
            numerator = (term1 + term2) % MOD
            
            ans = (numerator * p_s_inv) % MOD
            return ans
        else:  # s != t
            # Formula for s != t: ways = ( (n-1)^k - (-1)^k ) / p_s
            # We need (term1 - term2_sign) mod MOD
            numerator = (term1 - term2_sign + MOD) % MOD

            ans = (numerator * p_s_inv) % MOD
            return ans

    def _compute_pi(self, pattern: str) -> list[int]:
        """
        Computes the Knuth-Morris-Pratt (KMP) prefix function (pi table).
        pi[i] is the length of the longest proper prefix of pattern[0..i]
        which is also a suffix of pattern[0..i].
        """
        n = len(pattern)
        pi = [0] * n
        for i in range(1, n):
            j = pi[i - 1]
            while j > 0 and pattern[i] != pattern[j]:
                j = pi[j - 1]
            if pattern[i] == pattern[j]:
                j += 1
            pi[i] = j
        return pi