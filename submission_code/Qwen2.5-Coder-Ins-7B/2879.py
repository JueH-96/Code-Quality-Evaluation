class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                for l in range(i):
                    dp[i][j] = min(dp[i][j], dp[l][j - 1] + self.changes(s[l:i]))

        return dp[n][k]

    def changes(self, s: str) -> int:
        n = len(s)
        d = n // 2
        return sum(s[i] != s[n - 1 - i] for i in range(d))