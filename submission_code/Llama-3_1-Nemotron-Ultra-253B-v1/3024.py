class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        if n != len(t):
            return 0
        if not self.is_rotation(s, t):
            return 0
        d = self.minimal_period(s)
        m = n - 1
        if s == t:
            part1 = pow(m, k, MOD) * (n // d) % MOD
            part2 = pow(-1, k, MOD) * (n - (n // d)) % MOD
            numerator = (part1 + part2) % MOD
            ans = numerator * pow(n, MOD-2, MOD) % MOD
        else:
            numerator = (pow(m, k, MOD) - pow(-1, k, MOD)) % MOD
            ans = numerator * pow(d, MOD-2, MOD) % MOD
        return ans % MOD
    
    def is_rotation(self, s, t):
        return len(s) == len(t) and t in s + s
    
    def minimal_period(self, s):
        n = len(s)
        if n == 0:
            return 0
        fail = self.compute_failure(s)
        period = n - fail[-1]
        if n % period == 0:
            return period
        else:
            return n
    
    def compute_failure(self, s):
        n = len(s)
        fail = [0] * n
        j = 0
        for i in range(1, n):
            while j > 0 and s[i] != s[j]:
                j = fail[j-1]
            if s[i] == s[j]:
                j += 1
                fail[i] = j
            else:
                fail[i] = 0
        return fail