class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        
        row_ones = [sum(row) for row in grid]
        n_cols = len(grid[0])
        col_ones = [0] * n_cols
        
        for j in range(n_cols):
            for i in range(len(grid)):
                col_ones[j] += grid[i][j]
        
        total = 0
        for i in range(len(grid)):
            for j in range(n_cols):
                if grid[i][j] == 1:
                    total += (row_ones[i] - 1) * (col_ones[j] - 1)
        
        return total