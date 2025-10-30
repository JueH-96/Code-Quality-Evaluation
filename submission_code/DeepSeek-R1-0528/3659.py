class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        mod = 10**9 + 7
        m = len(grid)
        n = len(grid[0])
        dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]
        dp[0][0][grid[0][0]] = 1
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                if i > 0:
                    for x in range(16):
                        new_x = x ^ grid[i][j]
                        dp[i][j][new_x] = (dp[i][j][new_x] + dp[i-1][j][x]) % mod
                if j > 0:
                    for x in range(16):
                        new_x = x ^ grid[i][j]
                        dp[i][j][new_x] = (dp[i][j][new_x] + dp[i][j-1][x]) % mod
        
        return dp[m-1][n-1][k] % mod