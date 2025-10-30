class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        
        # Precompute the count of 1s in each row and column
        row_count = [sum(grid[i]) for i in range(n)]
        col_count = [sum(grid[i][j] for i in range(n)) for j in range(m)]
        
        count = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    # Number of 1s in the same row (excluding the current position)
                    row_ones = row_count[i] - 1
                    # Number of 1s in the same column (excluding the current position)
                    col_ones = col_count[j] - 1
                    
                    # Number of right triangles with the right angle at (i, j)
                    count += row_ones * col_ones
        
        return count