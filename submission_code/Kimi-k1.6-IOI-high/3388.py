class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        m, n = len(grid), len(grid[0])
        row_counts = [sum(row) for row in grid]
        col_counts = [sum(row[j] for row in grid) for j in range(n)]
        result = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    result += (row_counts[i] - 1) * (col_counts[j] - 1)
        return result