class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        """
        We consider each "operation" on s as rotating s to the left by some i (1 <= i < n).
        After k such rotations (where each rotation i_j is in [1..n-1]), the net effect
        is a rotation by (i_1 + i_2 + ... + i_k) mod n.

        So we need to count how many ways the sum of k chosen increments (each in [1..n-1])
        equals a particular rotation offset x mod n, for which r_x(s) = t.

        Key steps:
        1) Find all offsets x in [0..n-1] such that rotating s left by x yields t. This can be
           done by searching t in (s + s) (KMP or Z-alg), recording matches x < n.
        2) Let R be that set of valid offsets. If R is empty, answer is 0.
        3) We then need dp(k, x) := number of ways to pick k increments in [1..n-1] summing to x mod n.
           The total number of k-picks is (n-1)^k. One can show (via a simple recurrence or
           inclusion-exclusion argument) that for x=0:
               dp(k,0) = A_k = ( (n-1)^k + (n-1)*(-1)^k ) / n
           and for x != 0:
               dp(k,x) = B_k = ( (n-1)^k - (-1)^k ) / n
           (all operations modulo 10^9+7).

        4) Finally, if c0 = count of x in R that are 0, and cNon0 = count of x in R that are != 0,
           the answer = c0*A_k + cNon0*B_k (mod 10^9+7).

        We just need to implement these steps carefully, handling large k by fast exponentiation,
        and finding R via a linear-time string search.
        """

        MOD = 10**9 + 7
        n = len(s)

        # 1) Build KMP prefix-function for pattern = t
        pi = [0]*n
        j = 0
        # Build the prefix table for t
        for i in range(1, n):
            while j > 0 and t[i] != t[j]:
                j = pi[j-1]
            if t[i] == t[j]:
                j += 1
            pi[i] = j

        # 2) KMP search for t in s+s, record matches < n
        matches = []
        text = s + s
        m = 0  # matched length so far in pattern
        for i in range(2*n):
            while m > 0 and text[i] != t[m]:
                m = pi[m-1]
            if text[i] == t[m]:
                m += 1
            if m == n:
                # match ends at index i; match start = i - n + 1
                start = i - n + 1
                if start < n:
                    matches.append(start)
                m = pi[m-1]

        # If no valid rotation offsets, answer is 0
        if not matches:
            return 0

        # Count how many times offset 0 appears vs. non-zero
        c0 = sum(1 for x in matches if x == 0)
        cNon0 = len(matches) - c0

        # Fast exponent (n-1)^k mod MOD
        pow_n_minus_1 = pow(n-1, k, MOD)

        # Compute (-1)^k mod MOD (either 1 or MOD-1)
        neg1 = 1 if (k % 2 == 0) else (MOD - 1)

        # Modular inverse of n
        inv_n = pow(n, MOD-2, MOD)

        # A_k = dp(k,0) = ((n-1)^k + (n-1)*(-1)^k) / n  mod
        # B_k = dp(k,x != 0) = ((n-1)^k - (-1)^k) / n   mod
        # Do it carefully with mod arithmetic
        A_k_num = (pow_n_minus_1 + (n-1)*neg1) % MOD   # numerator of A_k
        A_k = (A_k_num * inv_n) % MOD
        B_k_num = (pow_n_minus_1 - neg1) % MOD         # numerator of B_k
        B_k = (B_k_num * inv_n) % MOD

        # Final answer
        ans = (c0 * A_k + cNon0 * B_k) % MOD
        return ans