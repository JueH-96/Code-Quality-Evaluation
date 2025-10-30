from typing import List

class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        rows = []
        cols = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    rows.append(i)
                    cols.append(j)
        min_row = min(rows)
        max_row = max(rows)
        min_col = min(cols)
        max_col = max(cols)
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        return height * width