class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        powers = []
        i = 1
        while True:
            current = i ** x
            if current > n:
                break
            powers.append(current)
            i += 1
        
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for p in powers:
            for j in range(n, p - 1, -1):
                dp[j] = (dp[j] + dp[j - p]) % MOD
        
        return dp[n] % MOD