from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        if rows == 0:
            return 0
        cols = len(grid[0])
        row_counts = [sum(row) for row in grid]
        col_counts = [0] * cols
        for i in range(rows):
            for j in range(cols):
                col_counts[j] += grid[i][j]
        res = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    res += (row_counts[i] - 1) * (col_counts[j] - 1)
        return res