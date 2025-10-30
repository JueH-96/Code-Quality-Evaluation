class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        
        # Precompute prefix sums for 'X' and 'Y'
        prefix_x = [[0] * (n + 1) for _ in range(m + 1)]
        prefix_y = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                prefix_x[i][j] = prefix_x[i-1][j] + prefix_x[i][j-1] - prefix_x[i-1][j-1] + (grid[i-1][j-1] == 'X')
                prefix_y[i][j] = prefix_y[i-1][j] + prefix_y[i][j-1] - prefix_y[i-1][j-1] + (grid[i-1][j-1] == 'Y')
        
        count = 0
        
        # Check if grid[0][0] is 'X' or 'Y'
        if grid[0][0] not in ['X', 'Y']:
            return 0
        
        # Iterate through all possible submatrices
        for i1 in range(m):
            for j1 in range(n):
                for i2 in range(i1, m):
                    for j2 in range(j1, n):
                        x_count = prefix_x[i2+1][j2+1] - prefix_x[i1][j2+1] - prefix_x[i2+1][j1] + prefix_x[i1][j1]
                        y_count = prefix_y[i2+1][j2+1] - prefix_y[i1][j2+1] - prefix_y[i2+1][j1] + prefix_y[i1][j1]
                        
                        if x_count == y_count and x_count > 0:
                            count += 1
        
        return count