from typing import List

class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        # Initialize boundaries
        r_min, r_max = m, -1
        c_min, c_max = n, -1
        
        # Scan the grid to find the min/max row and column indices of 1's
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    if i < r_min:
                        r_min = i
                    if i > r_max:
                        r_max = i
                    if j < c_min:
                        c_min = j
                    if j > c_max:
                        c_max = j
        
        # Compute height and width of the bounding rectangle
        height = r_max - r_min + 1
        width = c_max - c_min + 1
        
        return height * width