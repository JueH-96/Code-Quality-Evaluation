class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        if m == 0:
            return 0
        n = len(grid[0])
        if n == 0:
            return 0

        row_flips = 0
        for r in grid:
            for j in range(n // 2):
                if r[j] != r[n - 1 - j]:
                    row_flips += 1

        col_flips = 0
        for j in range(n):
            for i in range(m // 2):
                if grid[i][j] != grid[m - 1 - i][j]:
                    col_flips += 1

        return min(row_flips, col_flips)