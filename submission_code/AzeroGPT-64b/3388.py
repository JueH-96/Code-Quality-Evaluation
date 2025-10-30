class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])
        res = 0
        for r in range(row):
            for c in range(col):
                if grid[r][c] == 1:
                    for k in range(r + 1, row):
                        if grid[k][c] == 1:
                            res += sum(grid[r][i] == 1 and grid[k][i] == 1 for i in range(col))
                    for k in range(c + 1, col):
                        if grid[r][k] == 1:
                            res += sum(grid[i][c] == 1 and grid[i][k] == 1 for i in range(row))
                    # Remove count from square 1s (diagonals right and bottom)
                    for i in range(r + 1, row):
                        for j in range(c + 1, col):
                            if grid[r][j] == 1 and grid[i][c] == 1 and grid[i][j] == 1:
                                res -= 1

        return res