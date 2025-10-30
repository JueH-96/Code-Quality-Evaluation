from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0]) if m > 0 else 0
        
        # Initialize prefix sum arrays for X and Y
        prefix_x = [[0] * (n + 1) for _ in range(m + 1)]
        prefix_y = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m):
            for j in range(n):
                current_x = 1 if grid[i][j] == 'X' else 0
                current_y = 1 if grid[i][j] == 'Y' else 0
                # Update the prefix sums
                prefix_x[i+1][j+1] = current_x + prefix_x[i][j+1] + prefix_x[i+1][j] - prefix_x[i][j]
                prefix_y[i+1][j+1] = current_y + prefix_y[i][j+1] + prefix_y[i+1][j] - prefix_y[i][j]
        
        result = 0
        for i in range(m):
            for j in range(n):
                x_count = prefix_x[i+1][j+1]
                y_count = prefix_y[i+1][j+1]
                if x_count == y_count and x_count >= 1:
                    result += 1
        
        return result