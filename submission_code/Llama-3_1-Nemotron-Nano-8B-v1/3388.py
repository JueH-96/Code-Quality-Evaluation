from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        rows = len(grid)
        cols = len(grid[0])
        
        # Calculate the number of 1s in each row
        row_counts = [sum(row) for row in grid]
        
        # Calculate the number of 1s in each column
        col_counts = [0] * cols
        for j in range(cols):
            cnt = 0
            for i in range(rows):
                cnt += grid[i][j]
            col_counts[j] = cnt
        
        total = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    # Number of 1s in the row (excluding current cell)
                    r = row_counts[i] - 1
                    # Number of 1s in the column (excluding current cell)
                    c = col_counts[j] - 1
                    total += r * c
        
        return total