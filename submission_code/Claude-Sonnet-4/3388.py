class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        # Precompute row and column sums
        row_sums = [sum(row) for row in grid]
        col_sums = [sum(grid[i][j] for i in range(m)) for j in range(n)]
        
        count = 0
        
        # For each cell with value 1, consider it as the corner of right triangle
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    # Count other 1's in same row and column
                    row_ones = row_sums[i] - 1  # exclude current cell
                    col_ones = col_sums[j] - 1  # exclude current cell
                    
                    # Number of right triangles with (i,j) as corner
                    count += row_ones * col_ones
        
        return count