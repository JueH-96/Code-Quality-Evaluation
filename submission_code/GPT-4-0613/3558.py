class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        dp = [[0]*n for _ in range(m)]
        dp[0][0] = health - grid[0][0]
        for i in range(1, m):
            dp[i][0] = max(dp[i-1][0] - grid[i][0], 0)
        for j in range(1, n):
            dp[0][j] = max(dp[0][j-1] - grid[0][j], 0)
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = max(min(dp[i-1][j], dp[i][j-1]) - grid[i][j], 0)
        return dp[-1][-1] > 0