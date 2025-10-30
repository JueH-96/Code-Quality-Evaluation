from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m = len(grid)
        if m == 0:
            return 0
        n = len(grid[0])
        
        # Initialize prefix sum matrices for X and Y
        x_count = [[0] * n for _ in range(m)]
        y_count = [[0] * n for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                current_x = 1 if grid[i][j] == 'X' else 0
                current_y = 1 if grid[i][j] == 'Y' else 0
                
                if i == 0 and j == 0:
                    x_count[i][j] = current_x
                    y_count[i][j] = current_y
                elif i == 0:
                    x_count[i][j] = x_count[i][j-1] + current_x
                    y_count[i][j] = y_count[i][j-1] + current_y
                elif j == 0:
                    x_count[i][j] = x_count[i-1][j] + current_x
                    y_count[i][j] = y_count[i-1][j] + current_y
                else:
                    x_count[i][j] = x_count[i-1][j] + x_count[i][j-1] - x_count[i-1][j-1] + current_x
                    y_count[i][j] = y_count[i-1][j] + y_count[i][j-1] - y_count[i-1][j-1] + current_y
        
        result = 0
        for i in range(m):
            for j in range(n):
                x = x_count[i][j]
                y = y_count[i][j]
                if x == y and x >= 1:
                    result += 1
        
        return result