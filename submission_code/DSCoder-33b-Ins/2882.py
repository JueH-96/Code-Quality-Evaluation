class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        mod = 10**9 + 7
        dp = [0]*(n+1)
        dp[0] = 1
        for i in range(1, n+1):
            for j in range(n, i-1, -1):
                for k in range(1, j**(1/x)+1):
                    if j-k**x >= 0:
                        dp[j] = (dp[j] + dp[j-k**x]) % mod
        return dp[-1]