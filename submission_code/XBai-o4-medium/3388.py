from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        m = len(grid)
        if m == 0:
            return 0
        n = len(grid[0])
        
        # Calculate row counts of 1s
        row_counts = [sum(row) for row in grid]
        
        # Calculate column counts of 1s
        col_counts = [0] * n
        for i in range(m):
            for j in range(n):
                col_counts[j] += grid[i][j]
        
        total = 0
        
        # Calculate total number of right triangles
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    total += (row_counts[i] - 1) * (col_counts[j] - 1)
        
        return total