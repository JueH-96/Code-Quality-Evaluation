class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        mod = 10**9 + 7
        dp = [0]*(n+1)
        dp[0] = 1
        for i in range(1, n+1):
            for j in range(1, x+1):
                if i >= j:
                    dp[i] = (dp[i] + dp[i-j]) % mod
        return dp[n]