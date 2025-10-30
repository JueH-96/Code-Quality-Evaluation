MOD = 10 ** 9 + 7

class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        k %= len(s)
        s = list(s)
        t = list(t)
        
        if s == t and k == 0:
            ans = 0
            for i in range(1, len(s)):
                if all(s[(i + j) % len(s)] == s[j] for j in range(len(s))):
                    ans += 1
            return ans % MOD
        
        def z_func(s):
            n = len(s)
            l, r = 0, 0
            z = [0] * n
            for i in range(1, n):
                if i <= r:
                    z[i] = min(r - i + 1, z[i - l])
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
                if i + z[i] - 1 > r:
                    l, r = i, i + z[i] - 1
            return z
        
        z = z_func(t + t)
        
        s.reverse()
        t.reverse()
        
        zReverse = z_func(t + t)
        
        z, zReverse = z[len(t):], zReverse[len(t):]
        
        ans = 0
        for i in range(len(s)):
            if z[(len(s) - i + k) % len(s)] >= len(s) - i or \
                zReverse[(len(s) - i + k) % len(s)] >= len(s) - i:
                ans += 1
        return ans % MOD