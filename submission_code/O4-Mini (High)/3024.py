class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        # Quick checks
        if len(t) != n:
            return 0
        if k == 0:
            return 1 if s == t else 0

        # 1) Compute the minimal period p of s via KMP prefix-function
        pi = [0] * n
        j = 0
        for i in range(1, n):
            while j > 0 and s[i] != s[j]:
                j = pi[j-1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j
        p0 = n - pi[-1]
        p = p0 if n % p0 == 0 else n

        # 2) Check if t is a rotation of s by finding t in s+s
        ss = s + s
        L = ss.find(t)
        # If not found or shift ≥ n, impossible
        if L == -1 or L >= n:
            return 0

        # 3) Precompute a = (n-1)^k mod MOD, b = (-1)^k mod MOD
        a = pow(n - 1, k, MOD)
        b = 1 if (k & 1) == 0 else MOD - 1

        # 4) We must average over the subgroup of rotations of size p.
        #    If L==0 (t==s), sum over the coset H = {0, p, 2p, …}, which
        #    gives formula (a + (p-1)*b) / p.  Otherwise (t≠s),
        #    the coset has no 0 element and the sum is (a - b) / p.
        inv_p = pow(p, MOD - 2, MOD)
        if L == 0:
            numerator = (a + (p - 1) * b) % MOD
        else:
            numerator = (a - b) % MOD

        return (numerator * inv_p) % MOD