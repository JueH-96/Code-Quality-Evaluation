class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        max_val = int(n**(1/x)) + 1
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for i in range(1, max_val):
            power = i**x
            if power > n:
                break
            for j in range(n, power - 1, -1):
                dp[j] = (dp[j] + dp[j - power]) % MOD
        
        return dp[n]