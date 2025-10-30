from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        # Initialize prefix sum arrays
        prefix_diff = [[0] * n for _ in range(m)]
        prefix_X = [[0] * n for _ in range(m)]
        
        # Set the (0,0) cell
        if grid[0][0] == 'X':
            prefix_diff[0][0] = 1
            prefix_X[0][0] = 1
        elif grid[0][0] == 'Y':
            prefix_diff[0][0] = -1
            prefix_X[0][0] = 0
        else:  # '.'
            prefix_diff[0][0] = 0
            prefix_X[0][0] = 0
        
        # Fill the first row
        for j in range(1, n):
            diff_val = 1 if grid[0][j] == 'X' else -1 if grid[0][j] == 'Y' else 0
            x_val = 1 if grid[0][j] == 'X' else 0
            prefix_diff[0][j] = prefix_diff[0][j-1] + diff_val
            prefix_X[0][j] = prefix_X[0][j-1] + x_val
        
        # Fill the first column
        for i in range(1, m):
            diff_val = 1 if grid[i][0] == 'X' else -1 if grid[i][0] == 'Y' else 0
            x_val = 1 if grid[i][0] == 'X' else 0
            prefix_diff[i][0] = prefix_diff[i-1][0] + diff_val
            prefix_X[i][0] = prefix_X[i-1][0] + x_val
        
        # Fill the rest of the matrix
        for i in range(1, m):
            for j in range(1, n):
                diff_val = 1 if grid[i][j] == 'X' else -1 if grid[i][j] == 'Y' else 0
                x_val = 1 if grid[i][j] == 'X' else 0
                prefix_diff[i][j] = diff_val + prefix_diff[i-1][j] + prefix_diff[i][j-1] - prefix_diff[i-1][j-1]
                prefix_X[i][j] = x_val + prefix_X[i-1][j] + prefix_X[i][j-1] - prefix_X[i-1][j-1]
        
        # Count the number of submatrices that satisfy the conditions
        count = 0
        for i in range(m):
            for j in range(n):
                if prefix_diff[i][j] == 0 and prefix_X[i][j] >= 1:
                    count += 1
        
        return count