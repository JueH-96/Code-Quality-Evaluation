class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Check if the grid already contains a 2x2 square of the same color
        for i in range(2):
            for j in range(2):
                if grid[i][j] == grid[i][j+1] == grid[i+1][j] == grid[i+1][j+1]:
                    return True
        
        # Try changing one cell to see if a 2x2 square can be formed
        for i in range(3):
            for j in range(3):
                if grid[i][j] == 'B':
                    grid[i][j] = 'W'
                    if self.canMakeSquare(grid):
                        return True
                    grid[i][j] = 'B'
                else:
                    grid[i][j] = 'B'
                    if self.canMakeSquare(grid):
                        return True
                    grid[i][j] = 'W'
        
        return False