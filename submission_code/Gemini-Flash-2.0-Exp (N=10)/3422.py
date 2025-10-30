class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        if n == 1:
            return 1
        
        dp = [[0] * n for _ in range(k + 1)]
        
        for i in range(n):
            dp[0][i] = 1
        
        for t in range(1, k + 1):
            for i in range(n):
                if i == 0:
                    dp[t][i] = 1
                else:
                    dp[t][i] = (dp[t][i-1] + dp[t-1][i]) % MOD
        
        return dp[k][n-1]