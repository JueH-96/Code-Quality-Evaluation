class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        row_flips = 0
        for i in range(m):
            flips = 0
            for j in range(n // 2):
                if grid[i][j] != grid[i][n - 1 - j]:
                    flips += 1
            row_flips += min(flips, n - flips)
        col_flips = 0
        for j in range(n):
            flips = 0
            for i in range(m // 2):
                if grid[i][j] != grid[m - 1 - i][j]:
                    flips += 1
            col_flips += min(flips, m - flips)
        return min(row_flips, col_flips)