from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0]) if m > 0 else 0
        
        prefix_x = [[0]*n for _ in range(m)]
        prefix_y = [[0]*n for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                current_x = 1 if grid[i][j] == 'X' else 0
                current_y = 1 if grid[i][j] == 'Y' else 0
                
                # Calculate top contribution
                top_x = prefix_x[i-1][j] if i > 0 else 0
                top_y = prefix_y[i-1][j] if i > 0 else 0
                
                # Calculate left contribution
                left_x = prefix_x[i][j-1] if j > 0 else 0
                left_y = prefix_y[i][j-1] if j > 0 else 0
                
                # Calculate diagonal for inclusion in top and left
                diagonal_x = prefix_x[i-1][j-1] if (i > 0 and j > 0) else 0
                diagonal_y = prefix_y[i-1][j-1] if (i > 0 and j > 0) else 0
                
                prefix_x[i][j] = top_x + left_x - diagonal_x + current_x
                prefix_y[i][j] = top_y + left_y - diagonal_y + current_y
        
        result = 0
        for i in range(m):
            for j in range(n):
                if prefix_x[i][j] == prefix_y[i][j] and prefix_x[i][j] > 0:
                    result += 1
        
        return result