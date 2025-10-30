class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0]) if m > 0 else 0
        
        # Initialize prefix sum matrices for X and Y
        sumX = [[0] * (n + 1) for _ in range(m + 1)]
        sumY = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m):
            for j in range(n):
                cell = grid[i][j]
                x = 1 if cell == 'X' else 0
                y = 1 if cell == 'Y' else 0
                sumX[i+1][j+1] = x + sumX[i][j+1] + sumX[i+1][j] - sumX[i][j]
                sumY[i+1][j+1] = y + sumY[i][j+1] + sumY[i+1][j] - sumY[i][j]
        
        count = 0
        for i in range(m):
            for j in range(n):
                x_total = sumX[i+1][j+1]
                y_total = sumY[i+1][j+1]
                if x_total == y_total and x_total >= 1:
                    count += 1
        return count