class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        # Create prefix sum arrays for 'X' and 'Y'
        x_prefix = [[0] * (n+1) for _ in range(m+1)]
        y_prefix = [[0] * (n+1) for _ in range(m+1)]
        
        for i in range(m):
            for j in range(n):
                x_count = 1 if grid[i][j] == 'X' else 0
                y_count = 1 if grid[i][j] == 'Y' else 0
                
                x_prefix[i+1][j+1] = x_prefix[i+1][j] + x_prefix[i][j+1] - x_prefix[i][j] + x_count
                y_prefix[i+1][j+1] = y_prefix[i+1][j] + y_prefix[i][j+1] - y_prefix[i][j] + y_count
        
        count = 0
        
        # Check all possible submatrices that start at (0,0)
        for i in range(m):
            for j in range(n):
                x_count = x_prefix[i+1][j+1]
                y_count = y_prefix[i+1][j+1]
                
                # Check if conditions are met
                if x_count == y_count and x_count > 0:
                    count += 1
        
        return count