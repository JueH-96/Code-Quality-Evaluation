class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        powers = []
        m = 1
        while True:
            current = m ** x
            if current > n:
                break
            powers.append(current)
            m += 1
        
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for num in powers:
            for j in range(n, num - 1, -1):
                dp[j] = (dp[j] + dp[j - num]) % MOD
        
        return dp[n] % MOD