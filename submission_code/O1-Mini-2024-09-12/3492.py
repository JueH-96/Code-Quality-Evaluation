from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])
        prefix_x = [[0]*n for _ in range(m)]
        prefix_y = [[0]*n for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                cur_x = 1 if grid[i][j] == 'X' else 0
                cur_y = 1 if grid[i][j] == 'Y' else 0
                prefix_x[i][j] = cur_x
                prefix_y[i][j] = cur_y
                
                if i > 0:
                    prefix_x[i][j] += prefix_x[i-1][j]
                    prefix_y[i][j] += prefix_y[i-1][j]
                if j > 0:
                    prefix_x[i][j] += prefix_x[i][j-1]
                    prefix_y[i][j] += prefix_y[i][j-1]
                if i > 0 and j > 0:
                    prefix_x[i][j] -= prefix_x[i-1][j-1]
                    prefix_y[i][j] -= prefix_y[i-1][j-1]
        
        count = 0
        for i in range(m):
            for j in range(n):
                x = prefix_x[i][j]
                y = prefix_y[i][j]
                if x == y and x >= 1:
                    count += 1
        return count