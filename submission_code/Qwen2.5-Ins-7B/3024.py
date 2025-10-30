class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        dp = [0] * (k + 1)
        dp[0] = 1
        ways = [1] * n
        
        for i in range(1, n):
            if s[i] == s[i-1]:
                ways[i] = ways[i-1] * 2 % MOD
            else:
                ways[i] = (ways[i-1] * 2 - ways[i-1] + MOD) % MOD
        
        for i in range(1, n):
            if s[i] == t[i]:
                dp[i] = dp[i-1]
            else:
                dp[i] = (dp[i-1] * (ways[i-1] - 1) + dp[i-1]) % MOD
        
        for i in range(1, n):
            if s[0] == t[i]:
                dp[n-1] = (dp[n-1] + dp[n-i-2] * ways[n-i-1]) % MOD
            else:
                dp[n-1] = (dp[n-1] + dp[n-i-2] * (ways[n-i-1] - 1)) % MOD
        
        return dp[n-1]