class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        n = len(grid)
        m = len(grid[0])
        
        # Build prefix sums for counts of 'X' and 'Y'
        prefix_x = [[0]*(m+1) for _ in range(n+1)]
        prefix_y = [[0]*(m+1) for _ in range(n+1)]
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                prefix_x[i][j] = prefix_x[i-1][j] + prefix_x[i][j-1] - prefix_x[i-1][j-1] + (1 if grid[i-1][j-1] == 'X' else 0)
                prefix_y[i][j] = prefix_y[i-1][j] + prefix_y[i][j-1] - prefix_y[i-1][j-1] + (1 if grid[i-1][j-1] == 'Y' else 0)
        
        # Count submatrices that start from (0,0) and end at (i,j) with
        # equal number of X and Y and at least one X
        count = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                total_x = prefix_x[i][j]  # number of 'X' in submatrix (0,0)->(i-1,j-1)
                total_y = prefix_y[i][j]  # number of 'Y' in submatrix (0,0)->(i-1,j-1)
                if total_x == total_y and total_x > 0:
                    count += 1
        
        return count