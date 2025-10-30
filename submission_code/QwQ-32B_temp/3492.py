from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        rows = len(grid)
        if rows == 0:
            return 0
        cols = len(grid[0])
        
        x_prefix = [[0] * cols for _ in range(rows)]
        y_prefix = [[0] * cols for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                current_x = 1 if grid[i][j] == 'X' else 0
                current_y = 1 if grid[i][j] == 'Y' else 0
                
                if i == 0 and j == 0:
                    x_prefix[i][j] = current_x
                    y_prefix[i][j] = current_y
                elif i == 0:
                    x_prefix[i][j] = x_prefix[i][j-1] + current_x
                    y_prefix[i][j] = y_prefix[i][j-1] + current_y
                elif j == 0:
                    x_prefix[i][j] = x_prefix[i-1][j] + current_x
                    y_prefix[i][j] = y_prefix[i-1][j] + current_y
                else:
                    x_prefix[i][j] = x_prefix[i-1][j] + x_prefix[i][j-1] - x_prefix[i-1][j-1] + current_x
                    y_prefix[i][j] = y_prefix[i-1][j] + y_prefix[i][j-1] - y_prefix[i-1][j-1] + current_y
        
        result = 0
        for i in range(rows):
            for j in range(cols):
                x = x_prefix[i][j]
                y = y_prefix[i][j]
                if x == y and x >= 1:
                    result += 1
        return result