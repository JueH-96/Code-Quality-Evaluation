class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        rows = len(grid)
        if rows == 0:
            return 0
        cols = len(grid[0])
        x = [[0] * cols for _ in range(rows)]
        y = [[0] * cols for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                if i == 0 and j == 0:
                    x_val = 1 if grid[i][j] == 'X' else 0
                    y_val = 1 if grid[i][j] == 'Y' else 0
                elif i == 0:
                    x_val = x[i][j-1] + (1 if grid[i][j] == 'X' else 0)
                    y_val = y[i][j-1] + (1 if grid[i][j] == 'Y' else 0)
                elif j == 0:
                    x_val = x[i-1][j] + (1 if grid[i][j] == 'X' else 0)
                    y_val = y[i-1][j] + (1 if grid[i][j] == 'Y' else 0)
                else:
                    x_val = x[i-1][j] + x[i][j-1] - x[i-1][j-1] + (1 if grid[i][j] == 'X' else 0)
                    y_val = y[i-1][j] + y[i][j-1] - y[i-1][j-1] + (1 if grid[i][j] == 'Y' else 0)
                x[i][j] = x_val
                y[i][j] = y_val
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if x[i][j] == y[i][j] and x[i][j] >= 1:
                    count += 1
        return count