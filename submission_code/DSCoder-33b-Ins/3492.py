class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0]*n for _ in range(m)]
        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 'X':
                    dp[i][j] = dp[i][j-1] + 1 if j > 0 else 1
                    min_width = dp[i][j]
                    for k in range(i, -1, -1):
                        if grid[k][j] == 'Y':
                            min_width = min(min_width, dp[k][j])
                            res += min_width
                        else:
                            break
        return res