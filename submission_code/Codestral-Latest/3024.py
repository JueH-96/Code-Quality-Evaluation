class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7

        if s == t:
            return pow(k, len(s) - 1, MOD)

        n = len(s)
        s = s + s
        t_indices = [i for i in range(n) if t.startswith(s[i:i+n])]

        if not t_indices:
            return 0

        dp = [0] * n
        dp[0] = 1

        for _ in range(1, k + 1):
            new_dp = [0] * n
            for idx in t_indices:
                new_dp[idx] = (new_dp[idx] + dp[(idx - 1) % n]) % MOD
            dp = new_dp

        return sum(dp) % MOD