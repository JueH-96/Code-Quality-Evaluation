class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10 ** 9 + 7
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[1] = [1] * (k + 1)
        
        for i in range(2, n + 1):
            for j in range(k + 1):
                for l in range(1, m + 1):
                    if j > 0 and l == 1:
                        dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % MOD
                    elif l != 1:
                        dp[i][j] = (dp[i][j] + dp[i - 1][j]) % MOD
        
        return sum(dp[n]) % MOD