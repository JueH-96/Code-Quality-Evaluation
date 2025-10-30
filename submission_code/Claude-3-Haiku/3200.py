class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for i in range(1, n + 1):
            for j in range(max(0, i - 4), i):
                dp[i] = (dp[i] + dp[j]) % MOD
        
        return dp[n]