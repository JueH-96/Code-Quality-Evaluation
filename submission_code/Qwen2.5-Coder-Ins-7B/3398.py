class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def check_square(r, c):
            return grid[r][c] == grid[r+1][c] == grid[r][c+1] == grid[r+1][c+1]
        
        for i in range(2):
            for j in range(2):
                if check_square(i, j):
                    return True
                grid[i][j], grid[i][j+1], grid[i+1][j], grid[i+1][j+1] = grid[i+1][j+1], grid[i][j], grid[i+1][j+1], grid[i][j]
                if check_square(i, j):
                    return True
                grid[i][j], grid[i][j+1], grid[i+1][j], grid[i+1][j+1] = grid[i][j], grid[i][j+1], grid[i+1][j], grid[i+1][j+1]
        return False