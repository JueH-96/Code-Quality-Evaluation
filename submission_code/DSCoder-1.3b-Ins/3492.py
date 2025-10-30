class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0

        rows, cols = len(grid), len(grid[0])
        dp = [[0] * cols for _ in range(rows)]
        res = 0

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 'X':
                    if i > 0:
                        dp[i][j] = dp[i-1][j] + 1
                    else:
                        dp[i][j] = 1
                    res += dp[i][j]
                elif grid[i][j] == 'Y':
                    if i > 0 and j > 0:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + 1
                    elif i > 0:
                        dp[i][j] = dp[i-1][j] + 1
                    else:
                        dp[i][j] = dp[i][j-1] + 1 if j > 0 else 1
                    res += dp[i][j]

        return res