class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute all possible a^x where a^x <= n
        powers = []
        a = 1
        while True:
            power = a ** x
            if power > n:
                break
            powers.append(power)
            a += 1
        
        # Initialize dp array
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for p in powers:
            for i in range(n, p - 1, -1):
                dp[i] = (dp[i] + dp[i - p]) % MOD
        
        return dp[n]