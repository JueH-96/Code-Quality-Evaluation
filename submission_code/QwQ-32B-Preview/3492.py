from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        cumulative_X = [[0] * cols for _ in range(rows)]
        cumulative_Y = [[0] * cols for _ in range(rows)]
        
        # Initialize counts for the first cell
        cumulative_X[0][0] = 1 if grid[0][0] == 'X' else 0
        cumulative_Y[0][0] = 1 if grid[0][0] == 'Y' else 0
        
        # Fill the first row
        for j in range(1, cols):
            counts_X = 1 if grid[0][j] == 'X' else 0
            counts_Y = 1 if grid[0][j] == 'Y' else 0
            cumulative_X[0][j] = cumulative_X[0][j-1] + counts_X
            cumulative_Y[0][j] = cumulative_Y[0][j-1] + counts_Y
        
        # Fill the first column
        for i in range(1, rows):
            counts_X = 1 if grid[i][0] == 'X' else 0
            counts_Y = 1 if grid[i][0] == 'Y' else 0
            cumulative_X[i][0] = cumulative_X[i-1][0] + counts_X
            cumulative_Y[i][0] = cumulative_Y[i-1][0] + counts_Y
        
        # Fill the rest of the cumulative_X and cumulative_Y arrays
        for i in range(1, rows):
            for j in range(1, cols):
                counts_X = 1 if grid[i][j] == 'X' else 0
                counts_Y = 1 if grid[i][j] == 'Y' else 0
                cumulative_X[i][j] = cumulative_X[i-1][j] + cumulative_X[i][j-1] - cumulative_X[i-1][j-1] + counts_X
                cumulative_Y[i][j] = cumulative_Y[i-1][j] + cumulative_Y[i][j-1] - cumulative_Y[i-1][j-1] + counts_Y
        
        # Count the number of valid submatrices
        result = 0
        for i in range(rows):
            for j in range(cols):
                count_X = cumulative_X[i][j]
                count_Y = cumulative_Y[i][j]
                if count_X == count_Y and count_X >= 1:
                    result += 1
        return result