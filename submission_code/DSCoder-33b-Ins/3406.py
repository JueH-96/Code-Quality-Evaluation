class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        mod = 10**9 + 7
        dp = [[0]*(limit*2+2) for _ in range(zero+one+1)]
        dp[0][zero] = 1
        for i in range(zero+one):
            for j in range(limit*2+1):
                if j-1 >= 0:
                    dp[i+1][j-1] = (dp[i+1][j-1] + dp[i][j]) % mod
                if j+1 <= limit*2:
                    dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j]*(i+1)) % mod
        return sum(dp[zero+one]) % mod