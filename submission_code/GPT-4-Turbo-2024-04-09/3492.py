class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        from collections import defaultdict
        
        if not grid or not grid[0] or grid[0][0] != 'X':
            return 0
        
        rows, cols = len(grid), len(grid[0])
        count = 0
        
        # Precompute prefix sums for 'X' and 'Y'
        prefix_x = [[0] * (cols + 1) for _ in range(rows + 1)]
        prefix_y = [[0] * (cols + 1) for _ in range(rows + 1)]
        
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                prefix_x[i][j] = prefix_x[i-1][j] + prefix_x[i][j-1] - prefix_x[i-1][j-1] + (1 if grid[i-1][j-1] == 'X' else 0)
                prefix_y[i][j] = prefix_y[i-1][j] + prefix_y[i][j-1] - prefix_y[i-1][j-1] + (1 if grid[i-1][j-1] == 'Y' else 0)
        
        # Check all submatrices starting from (0, 0)
        for end_row in range(1, rows + 1):
            for end_col in range(1, cols + 1):
                # Submatrix from (1,1) to (end_row, end_col)
                num_x = prefix_x[end_row][end_col] - prefix_x[0][end_col] - prefix_x[end_row][0] + prefix_x[0][0]
                num_y = prefix_y[end_row][end_col] - prefix_y[0][end_col] - prefix_y[end_row][0] + prefix_y[0][0]
                
                if num_x > 0 and num_x == num_y:
                    count += 1
        
        return count