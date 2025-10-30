from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        # Count number of 1s in each row and each column
        n = len(grid)
        m = len(grid[0])
        row_count = [sum(grid[i][j] for j in range(m)) for i in range(n)]
        col_count = [sum(grid[i][j] for i in range(n)) for j in range(m)]
        
        # For each 1-cell, the number of right triangles with right angle at (i, j)
        # is (number of other 1s in its row) * (number of other 1s in its column).
        result = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    result += (row_count[i] - 1) * (col_count[j] - 1)
        
        return result