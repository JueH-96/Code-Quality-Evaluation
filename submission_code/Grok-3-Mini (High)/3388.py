class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        
        # Compute the sum of 1's in each row
        row_sum = [sum(row) for row in grid]
        
        # Compute the sum of 1's in each column
        col_sum = [sum(grid[i][j] for i in range(rows)) for j in range(cols)]
        
        ans = 0
        
        # Iterate through each cell in the grid
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    # Add the number of right triangles with right angle at (i, j)
                    ans += (row_sum[i] - 1) * (col_sum[j] - 1)
        
        return ans