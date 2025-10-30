class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        # Precompute the prefix sums for 'X' and 'Y'
        prefix_X = [[0] * (cols + 1) for _ in range(rows + 1)]
        prefix_Y = [[0] * (cols + 1) for _ in range(rows + 1)]
        
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                prefix_X[i][j] = prefix_X[i-1][j] + prefix_X[i][j-1] - prefix_X[i-1][j-1] + (1 if grid[i-1][j-1] == 'X' else 0)
                prefix_Y[i][j] = prefix_Y[i-1][j] + prefix_Y[i][j-1] - prefix_Y[i-1][j-1] + (1 if grid[i-1][j-1] == 'Y' else 0)
        
        count = 0
        
        # Iterate over all possible submatrices that include grid[0][0]
        for i1 in range(1, rows + 1):
            for j1 in range(1, cols + 1):
                # The submatrix must include (0,0), so i2 >= i1 and j2 >= j1
                for i2 in range(i1, rows + 1):
                    for j2 in range(j1, cols + 1):
                        # Calculate the number of 'X' and 'Y' in the submatrix
                        x_count = prefix_X[i2][j2] - prefix_X[i1-1][j2] - prefix_X[i2][j1-1] + prefix_X[i1-1][j1-1]
                        y_count = prefix_Y[i2][j2] - prefix_Y[i1-1][j2] - prefix_Y[i2][j1-1] + prefix_Y[i1-1][j1-1]
                        if x_count == y_count and x_count > 0:
                            count += 1
        
        return count