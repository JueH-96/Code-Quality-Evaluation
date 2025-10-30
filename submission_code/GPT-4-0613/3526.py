class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        min_flips = float('inf')

        # Check rows
        row_flips = 0
        for row in grid:
            row_flips += min(row.count(0), row.count(1))
        min_flips = min(min_flips, row_flips)

        # Check columns
        col_flips = 0
        for j in range(n):
            col = [grid[i][j] for i in range(m)]
            col_flips += min(col.count(0), col.count(1))
        min_flips = min(min_flips, col_flips)

        return min_flips