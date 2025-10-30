class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        """
        We want to find the number of ways to transform s into t in exactly k operations,
        where each operation removes a non-empty, proper suffix of s and appends it to
        the front. Equivalently, each operation adds a non-zero "shift" (mod n) to s.

        Let n = len(s). Each operation chooses an i in [1..n-1], turning the current shift d
        into (d + i) mod n. After k operations, we want s^(final_shift) = t, i.e., s^D = t
        for final_shift = D. We count how many sequences (i_1, ..., i_k) yield final_shift D
        mod n, then sum over all D with s^D = t.

        Key facts:
          1) The number of ways to get from shift 0 to shift D in k steps, allowing i_j in
             [1..n-1], is given by the (0, D) entry of the matrix M^k, where M is the n×n
             matrix with M[a, b] = 1 if b != a, else 0. This M = J - I (J is all 1s, I is identity).
          2) M has eigenvalues (n-1) (with multiplicity 1) and -1 (with multiplicity n-1).
             From this, one can deduce the closed-form for (M^k)[0, D]:
               (M^k)[0, 0] = (1/n) * ( (n-1)^k + (n-1)*(-1)^k )
               (M^k)[0, D] = (1/n) * ( (n-1)^k - (-1)^k )  for D != 0
          3) We let ValidD = { D | s^D = t }. Let c = |ValidD|. Also note if s == t then 0 ∈ ValidD.
             - If c = 0, answer = 0.
             - Otherwise, compute:
                  p = (n-1)^k (mod), m1p = (-1)^k (mod).  (Here, (-1)^k is 1 if k even, or -1 mod.)
                  Let invn = the modular inverse of n.
                  Then:
                    · If 0 ∈ ValidD (i.e. s == t), final answer = (1/n) * [ c·(n-1)^k + (n-c)*(-1)^k ] mod
                    · Else, final answer = (1/n) * [ c·( (n-1)^k - (-1)^k ) ] mod.

        Steps:
          1) Build s+s, run KMP (or Z-alg) to count how many shifts D in [0..n-1] match t.
             That count is c. Also note if shift 0 is included (c0).
          2) Compute powers mod 10^9+7.
          3) Apply formula.

        This runs in O(n) for the KMP plus O(log k) for exponentiation, which is efficient for
        n up to 5e5 and k up to 1e15.
        """

        mod = 10**9 + 7
        n = len(s)

        # Quick check: if lengths differ (shouldn't per problem statement), return 0
        if len(t) != n:
            return 0

        # 1) KMP to find all occurrences of t in s+s (only up to index < n)
        def buildLPS(p):
            m = len(p)
            lps = [0]*m
            j = 0
            for i in range(1,m):
                while j > 0 and p[i] != p[j]:
                    j = lps[j-1]
                if p[i] == p[j]:
                    j += 1
                    lps[i] = j
            return lps

        def countMatches(s, t):
            s2 = s + s  # length = 2n
            lps = buildLPS(t)
            j = 0
            count_all = 0
            found_zero = 0
            for i in range(len(s2)):
                while j > 0 and s2[i] != t[j]:
                    j = lps[j-1]
                if s2[i] == t[j]:
                    j += 1
                    if j == len(t):
                        start_idx = i - len(t) + 1
                        if start_idx < n:
                            count_all += 1
                            if start_idx == 0:
                                found_zero = 1
                        j = lps[j-1]
            return count_all, found_zero

        c, c0 = countMatches(s, t)
        if c == 0:
            return 0  # No way to match t by any rotation.

        # 2) Compute the powers:
        #   p = (n-1)^k mod
        #   m1p = (-1)^k mod (which is 1 if k is even, else mod-1)
        p = pow(n-1, k, mod)
        if k % 2 == 0:
            m1p = 1
        else:
            m1p = mod - 1  # = -1 mod

        # Modular inverse of n (n <= 5*10^5, which is coprime to mod=1e9+7)
        invn = pow(n, mod-2, mod)

        # 3) Apply the formula
        if c0 == 1:
            # s == t implies shift=0 is valid
            # answer = (1/n) * [ c*(n-1)^k + (n-c)*(-1)^k ]
            val = (c * p) % mod
            val = (val + (n - c) * m1p) % mod
            ans = (val * invn) % mod
            return ans
        else:
            # answer = (1/n) * [ c * ( (n-1)^k - (-1)^k ) ]
            diff = (p - m1p) % mod
            val = (c * diff) % mod
            ans = (val * invn) % mod
            return ans