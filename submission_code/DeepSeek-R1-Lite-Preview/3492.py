class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        # Initialize prefix sums for 'X' and 'Y' with an extra row and column of zeros
        prefix_X = [[0] * (cols + 1) for _ in range(rows + 1)]
        prefix_Y = [[0] * (cols + 1) for _ in range(rows + 1)]
        
        # Compute prefix sums for 'X' and 'Y'
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                prefix_X[i][j] = prefix_X[i-1][j] + prefix_X[i][j-1] - prefix_X[i-1][j-1] + (1 if grid[i-1][j-1] == 'X' else 0)
                prefix_Y[i][j] = prefix_Y[i-1][j] + prefix_Y[i][j-1] - prefix_Y[i-1][j-1] + (1 if grid[i-1][j-1] == 'Y' else 0)
        
        result = 0
        # Iterate through all possible submatrices starting from (0,0)
        for i in range(rows):
            for j in range(cols):
                count_X = prefix_X[i+1][j+1]
                count_Y = prefix_Y[i+1][j+1]
                if count_X == count_Y and count_X >= 1:
                    result += 1
        return result