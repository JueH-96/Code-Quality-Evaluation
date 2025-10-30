class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        if rows == 0:
            return 0
        cols = len(grid[0])
        row_counts = [sum(row) for row in grid]
        column_counts = [sum(grid[i][j] for i in range(rows)) for j in range(cols)]
        total = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    total += (row_counts[i] - 1) * (column_counts[j] - 1)
        return total