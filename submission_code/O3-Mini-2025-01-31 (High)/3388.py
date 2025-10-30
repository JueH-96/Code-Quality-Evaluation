from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0]) if m > 0 else 0
        
        # Pre-compute the number of ones in each row and each column.
        rowCounts = [sum(row) for row in grid]
        colCounts = [sum(grid[i][j] for i in range(m)) for j in range(n)]
        
        count = 0
        # For each cell with a 1, count the pairs from its row and column (excluding itself)
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    row_ones = rowCounts[i] - 1  # other ones in the same row
                    col_ones = colCounts[j] - 1  # other ones in the same column
                    count += row_ones * col_ones
        return count