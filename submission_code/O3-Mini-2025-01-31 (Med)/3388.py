from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        # Precompute count of ones per row and per column.
        rowCount = [0] * m
        colCount = [0] * n
        
        # Count ones in rows and columns
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    rowCount[i] += 1
                    colCount[j] += 1
        
        res = 0
        # For each cell that is 1, consider it as the vertex of the right angle.
        # The triangle can be formed with any one from same row (excluding itself)
        # and any one from same column (excluding itself).
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    # Number of valid choices in the row and column respectively.
                    choices_row = rowCount[i] - 1
                    choices_col = colCount[j] - 1
                    if choices_row > 0 and choices_col > 0:
                        res += choices_row * choices_col
        return res