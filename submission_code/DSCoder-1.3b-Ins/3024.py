class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + (s[i] == 'a')
            suffix[i + 1] = suffix[i] + (s[i] == 'a')
        mod = 10**9 + 7
        ans = 0
        for i in range(n):
            if s[i] == 'a':
                ans = (ans + prefix[i] * (suffix[i] - i) * (k - 1)) % mod
                ans = (ans + (prefix[i] - i) * suffix[i] * (k - 2)) % mod
            else:
                ans = (ans + (prefix[i] - i) * suffix[i] * (k - 1)) % mod
                ans = (ans + suffix[i] * (suffix[i] - i) * (k - 2)) % mod
        return ans