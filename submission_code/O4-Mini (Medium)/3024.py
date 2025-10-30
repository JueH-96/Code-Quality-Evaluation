class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        # find rotation offset d such that rotating s right by d gives t
        # search t in s+s, only positions < n are valid rotations
        dbl = s + s
        d = dbl.find(t)
        if d == -1 or d >= n:
            return 0

        # compute minimal period of s
        # prefix-function (KMP) to get pi[-1]
        pi = [0] * n
        j = 0
        for i in range(1, n):
            while j > 0 and s[i] != s[j]:
                j = pi[j-1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j
        p = n - pi[n-1]
        per = p if n % p == 0 else n
        m = n // per  # number of distinct rotations that map s to itself

        # fast exponentiation
        pow_n1 = pow(n - 1, k, MOD)
        # b = (-1)^k mod MOD
        b = 1 if (k % 2 == 0) else (MOD - 1)
        # a = ((n-1)^k - (-1)^k) / n  (mod MOD)
        inv_n = pow(n, MOD - 2, MOD)
        a = (pow_n1 - b) * inv_n % MOD

        # if d == 0, t == s, we have one of the target residues = 0
        # answer = m * a + b  else answer = m * a
        if d == 0:
            ans = (m * a + b) % MOD
        else:
            ans = (m * a) % MOD

        return ans