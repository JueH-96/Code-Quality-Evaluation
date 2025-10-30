from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        rows = len(grid)
        if rows == 0:
            return 0
        cols = len(grid[0])
        prefix_x = [[0] * (cols + 1) for _ in range(rows + 1)]
        prefix_y = [[0] * (cols + 1) for _ in range(rows + 1)]
        
        for i in range(rows):
            for j in range(cols):
                # Update prefix_x[i+1][j+1]
                prefix_x[i+1][j+1] = prefix_x[i][j+1] + prefix_x[i+1][j] - prefix_x[i][j]
                if grid[i][j] == 'X':
                    prefix_x[i+1][j+1] += 1
                
                # Update prefix_y[i+1][j+1]
                prefix_y[i+1][j+1] = prefix_y[i][j+1] + prefix_y[i+1][j] - prefix_y[i][j]
                if grid[i][j] == 'Y':
                    prefix_y[i+1][j+1] += 1
        
        count = 0
        for k in range(rows):
            for l in range(cols):
                x = prefix_x[k+1][l+1]
                y = prefix_y[k+1][l+1]
                if x == y and x >= 1:
                    count += 1
        return count