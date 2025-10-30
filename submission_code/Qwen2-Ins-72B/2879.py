class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        dp = [[float('inf')] * k for _ in range(n)]
        cost = [[0] * n for _ in range(n)]
        
        for length in range(1, n + 1):
            for start in range(n - length + 1):
                end = start + length - 1
                cost[start][end] = cost[start + 1][end - 1] + (s[start] != s[end]) if length > 1 else 0
        
        for i in range(n):
            dp[i][0] = cost[0][i]
        
        for j in range(1, k):
            for i in range(j, n):
                for p in range(j - 1, i):
                    dp[i][j] = min(dp[i][j], dp[p][j - 1] + cost[p + 1][i])
        
        return dp[n - 1][k - 1]