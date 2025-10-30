class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        dp = [[float('inf')] * 100 for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(n):
            for j in range(100):
                dp[i + 1][j] = min(dp[i + 1][j], dp[i][j] + 1)
                dp[i + 1][(j * 10 + int(num[i])) % 100] = min(dp[i + 1][(j * 10 + int(num[i])) % 100], dp[i][j])
        return dp[n][25]