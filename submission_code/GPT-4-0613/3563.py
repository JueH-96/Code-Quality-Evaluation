class Solution:
    def maxScore(self, grid):
        n = len(grid)
        m = len(grid[0])
        dp = [[0]*1024 for _ in range(1024)]
        for i in range(n-1, -1, -1):
            for mask in range(1024):
                for j in range(m):
                    if (mask>>j)&1:
                        dp[i][mask] = max(dp[i][mask], grid[i][j] + (dp[i+1][mask^(1<<j)] if i+1<n else 0))
                    else:
                        dp[i][mask] = max(dp[i][mask], dp[i+1][mask] if i+1<n else 0)
        return dp[0][1023]