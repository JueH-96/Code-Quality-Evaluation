from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        dp = [[0]*n for _ in range(m)]
        dp[-1][-1] = health - grid[-1][-1]

        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                if i == m-1 and j == n-1:
                    continue
                right = dp[i][j+1] - grid[i][j] if j+1 < n else 0
                down = dp[i+1][j] - grid[i][j] if i+1 < m else 0
                dp[i][j] = max(right, down, 1)

        return dp[0][0] >= 1