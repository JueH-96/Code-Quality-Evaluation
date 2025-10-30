class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        # Count 1s in each row and column
        row_count = [0] * m
        col_count = [0] * n
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    row_count[i] += 1
                    col_count[j] += 1
        
        # Count right triangles
        result = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    # Number of right triangles with (i, j) as corner
                    result += (row_count[i] - 1) * (col_count[j] - 1)
        
        return result