class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        count = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    row_count = 0
                    col_count = 0
                    for k in range(cols):
                        if k != j and grid[i][k] == 1:
                            row_count += 1
                    for k in range(rows):
                        if k != i and grid[k][j] == 1:
                            col_count += 1
                    count += row_count * col_count
        return count