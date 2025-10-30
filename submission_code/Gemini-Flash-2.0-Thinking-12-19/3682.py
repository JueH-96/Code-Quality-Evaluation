class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        if k > n - 1:
            return 0
        if n == 1:
            return m if k == 0 else 0
        MOD = 10**9 + 7
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        for j in range(m):
            dp[1][0] = m
            
        dp[1][0] = m
        for i in range(2, n + 1):
            for j in range(k + 1):
                term1 = 0
                if j >= 1:
                    term1 = dp[i-1][j-1]
                term2 = (m - 1) * dp[i-1][j]
                dp[i][j] = (term1 + term2) % MOD
                
        return dp[n][k]