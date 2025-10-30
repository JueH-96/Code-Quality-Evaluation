class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        mod = 10**9 + 7
        dp = [0] * (n + 1)
        dp[0] = 1
        i = 1
        while True:
            v = i ** x
            if v > n:
                break
            for j in range(n, v - 1, -1):
                if j - v >= 0:
                    dp[j] = (dp[j] + dp[j - v]) % mod
            i += 1
        return dp[n] % mod