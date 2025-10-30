class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        
        # Create prefix sum arrays for 'X' and 'Y' counts
        prefix_x = [[0] * (n + 1) for _ in range(m + 1)]
        prefix_y = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Build prefix sums
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                x_count = 1 if grid[i-1][j-1] == 'X' else 0
                y_count = 1 if grid[i-1][j-1] == 'Y' else 0
                
                prefix_x[i][j] = prefix_x[i-1][j] + prefix_x[i][j-1] - prefix_x[i-1][j-1] + x_count
                prefix_y[i][j] = prefix_y[i-1][j] + prefix_y[i][j-1] - prefix_y[i-1][j-1] + y_count
        
        count = 0
        
        # Check all submatrices starting from (0,0)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Count of X's and Y's in submatrix from (0,0) to (i-1,j-1)
                x_count = prefix_x[i][j]
                y_count = prefix_y[i][j]
                
                # Check conditions: equal frequency and at least one X
                if x_count == y_count and x_count > 0:
                    count += 1
        
        return count