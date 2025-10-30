from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        count = 0
        
        # Precompute the number of 1s in each row and column up to each index
        row_counts = [[0] * cols for _ in range(rows)]
        col_counts = [[0] * cols for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    row_counts[i][j] = (row_counts[i][j-1] if j > 0 else 0) + 1
                    col_counts[i][j] = (col_counts[i-1][j] if i > 0 else 0) + 1
        
        # Count the number of right triangles
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    for k in range(j):
                        if grid[i][k] == 1:
                            count += col_counts[i-1][k] if i > 0 else 0
                    for k in range(i):
                        if grid[k][j] == 1:
                            count += row_counts[k][j-1] if j > 0 else 0
        
        return count