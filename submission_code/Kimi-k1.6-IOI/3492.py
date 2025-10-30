class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        prefix_X = [[0] * cols for _ in range(rows)]
        prefix_Y = [[0] * cols for _ in range(rows)]
        
        # Initialize the first cell
        prefix_X[0][0] = 1 if grid[0][0] == 'X' else 0
        prefix_Y[0][0] = 1 if grid[0][0] == 'Y' else 0
        
        # Fill first row for X and Y
        for j in range(1, cols):
            prefix_X[0][j] = prefix_X[0][j-1] + (1 if grid[0][j] == 'X' else 0)
            prefix_Y[0][j] = prefix_Y[0][j-1] + (1 if grid[0][j] == 'Y' else 0)
        
        # Fill first column for X and Y
        for i in range(1, rows):
            prefix_X[i][0] = prefix_X[i-1][0] + (1 if grid[i][0] == 'X' else 0)
            prefix_Y[i][0] = prefix_Y[i-1][0] + (1 if grid[i][0] == 'Y' else 0)
        
        # Fill the rest of the matrix
        for i in range(1, rows):
            for j in range(1, cols):
                current_X = 1 if grid[i][j] == 'X' else 0
                current_Y = 1 if grid[i][j] == 'Y' else 0
                prefix_X[i][j] = current_X + prefix_X[i-1][j] + prefix_X[i][j-1] - prefix_X[i-1][j-1]
                prefix_Y[i][j] = current_Y + prefix_Y[i-1][j] + prefix_Y[i][j-1] - prefix_Y[i-1][j-1]
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                x = prefix_X[i][j]
                y = prefix_Y[i][j]
                if x == y and x >= 1:
                    count += 1
        return count