class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        # Initialize boundaries to extreme values
        m, n = len(grid), len(grid[0])
        row_min, row_max = m, -1
        col_min, col_max = n, -1
        
        # Iterate over the grid to find the minimum and maximum row and column indices that contain a 1.
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    row_min = min(row_min, i)
                    row_max = max(row_max, i)
                    col_min = min(col_min, j)
                    col_max = max(col_max, j)
        
        # Compute the area of the smallest rectangle that encloses all the 1's.
        height = row_max - row_min + 1
        width = col_max - col_min + 1
        return height * width