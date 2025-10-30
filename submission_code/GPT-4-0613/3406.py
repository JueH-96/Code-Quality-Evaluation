class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        dp = [[[0]*(zero+1) for _ in range(one+1)] for _ in range(one+1)]
        dp[0][0][0] = 1
        for i in range(one+1):
            for j in range(i+1):
                for k in range(zero+1):
                    if dp[i][j][k] == 0:
                        continue
                    if i+1 <= one:
                        dp[i+1][j+1][max(0, k-1)] += dp[i][j][k]
                        dp[i+1][j+1][max(0, k-1)] %= MOD
                    if j and k+1 <= zero:
                        dp[i][j][k+1] += dp[i][j][k] * j
                        dp[i][j][k+1] %= MOD
        ans = 0
        for i in range(1, limit+1):
            for j in range(zero+1):
                ans += dp[one][i][j]
                ans %= MOD
        return ans