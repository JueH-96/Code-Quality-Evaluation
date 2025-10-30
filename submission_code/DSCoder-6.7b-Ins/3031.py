class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, m = len(grid), len(grid[0])
        rows, cols = [0]*n, [0]*m
        rows[0] = cols[0] = 1
        for i in range(1, n):
            rows[i] = (rows[i-1]*grid[i-1][0])%12345
        for j in range(1, m):
            cols[j] = (cols[j-1]*grid[0][j-1])%12345
        for i in range(1, n):
            for j in range(1, m):
                grid[i][j] = (grid[i-1][j]*grid[i][j-1])%12345
        for i in range(n-2, -1, -1):
            grid[i][m-1] = (grid[i+1][m-1]*grid[i][m-2])%12345
        for j in range(m-2, -1, -1):
            grid[n-1][j] = (grid[n-1][j+1]*grid[n-2][j])%12345
        for i in range(n-2, -1, -1):
            for j in range(m-2, -1, -1):
                grid[i][j] = (grid[i+1][j]*grid[i][j+1]*rows[i]*cols[j])%12345
        return grid