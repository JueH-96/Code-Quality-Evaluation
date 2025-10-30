class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        powers = []
        i = 1
        while True:
            temp = i ** x
            if temp > n:
                break
            powers.append(temp)
            i += 1
        
        dp = [0] * (n + 1)
        dp[0] = 1  # Base case: one way to make sum 0 (choose nothing)
        
        for num in powers:
            # Update dp in reverse to avoid reusing the same number multiple times
            for s in range(n, num - 1, -1):
                dp[s] = (dp[s] + dp[s - num]) % MOD
        
        return dp[n] % MOD