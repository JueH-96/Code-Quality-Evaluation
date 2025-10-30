class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        dp = [0] * (n + 1)
        dp[0] = 1
        mod = 10**9 + 7
        
        for i in range(1, n + 1):
            power = i ** x
            if power > n:
                break
            for j in range(n, power - 1, -1):
                dp[j] = (dp[j] + dp[j - power]) % mod
        
        return dp[n]