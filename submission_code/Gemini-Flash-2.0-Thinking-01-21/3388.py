class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    row_ones = 0
                    for k in range(cols):
                        if k != c and grid[r][k] == 1:
                            row_ones += 1
                    col_ones = 0
                    for k in range(rows):
                        if k != r and grid[k][c] == 1:
                            col_ones += 1
                    count += row_ones * col_ones
        return count