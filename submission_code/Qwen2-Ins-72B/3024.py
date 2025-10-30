class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        if s == t and k == 0:
            return 1
        if s == t and k % 2 == 0:
            return pow(n, k, MOD) * pow(pow(n - 1, MOD - 2, MOD), k, MOD) % MOD
        if s == t:
            return 0
        if k == 1:
            for i in range(n):
                if s[i:] + s[:i] == t:
                    return 1
            return 0
        if k % 2 == 0:
            return pow(n, k, MOD) * pow(pow(n - 1, MOD - 2, MOD), k, MOD) % MOD * self.numberOfWays(s, t, 0) % MOD
        return 0