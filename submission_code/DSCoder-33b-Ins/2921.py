class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        dp = [[0]*10 for _ in range(101)]
        for i in range(10):
            dp[1][i] = 1
        for i in range(2, 101):
            for j in range(10):
                if j > 0:
                    dp[i][j] = (dp[i][j] + dp[i-1][j-1]) % MOD
                if j < 9:
                    dp[i][j] = (dp[i][j] + dp[i-1][j+1]) % MOD
        prefix_dp = [[0]*10 for _ in range(101)]
        for i in range(10):
            prefix_dp[1][i] = dp[1][i]
        for i in range(2, 101):
            for j in range(10):
                prefix_dp[i][j] = (prefix_dp[i][j] + prefix_dp[i-1][j] + dp[i][j]) % MOD
        def solve(s: str) -> int:
            n = len(s)
            ans = 0
            for i in range(1, n):
                for j in range(1, 10):
                    ans = (ans + dp[i][j]) % MOD
            for i in range(1, int(s[0])):
                ans = (ans + dp[n][i]) % MOD
            for i in range(1, n):
                for j in range(int(s[i-1]), int(s[i])):
                    if abs(j-int(s[i-1])) != 1:
                        continue
                    ans = (ans + prefix_dp[n-i][j]) % MOD
                    if i == n-1 and j == int(s[i]) and int(s[i-1]) != j-1 and j != 0:
                        ans = (ans + 1) % MOD
            return ans
        return (solve(high) - solve(str(int(low)-1)) + MOD) % MOD