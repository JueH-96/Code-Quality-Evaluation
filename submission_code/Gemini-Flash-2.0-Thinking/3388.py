from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        count = 0

        row_counts = [0] * rows
        col_counts = [0] * cols

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    row_counts[r] += 1
                    col_counts[c] += 1

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    row_ones = row_counts[r] - 1
                    col_ones = col_counts[c] - 1
                    count += row_ones * col_ones

        return count