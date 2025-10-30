class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        R = len(grid)
        if R == 0:
            return 0
        C = len(grid[0])
        if C == 0:
            return 0
        
        # Initialize prefix sums for X and Y
        prefix_x = [[0] * (C + 1) for _ in range(R + 1)]
        prefix_y = [[0] * (C + 1) for _ in range(R + 1)]
        
        for i in range(1, R + 1):
            for j in range(1, C + 1):
                cell = grid[i-1][j-1]
                add_x = 1 if cell == 'X' else 0
                add_y = 1 if cell == 'Y' else 0
                prefix_x[i][j] = prefix_x[i-1][j] + prefix_x[i][j-1] - prefix_x[i-1][j-1] + add_x
                prefix_y[i][j] = prefix_y[i-1][j] + prefix_y[i][j-1] - prefix_y[i-1][j-1] + add_y
        
        result = 0
        for i in range(R):
            for j in range(C):
                x = prefix_x[i+1][j+1]
                y = prefix_y[i+1][j+1]
                if x == y and x >= 1:
                    result += 1
        return result