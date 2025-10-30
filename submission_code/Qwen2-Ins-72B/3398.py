class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Check all 2x2 squares in the grid
        for i in range(2):
            for j in range(2):
                # Check if all elements in the 2x2 square are the same color
                if grid[i][j] == grid[i][j+1] == grid[i+1][j] == grid[i+1][j+1]:
                    return True
        # Check if changing one cell can make a 2x2 square of the same color
        for i in range(3):
            for j in range(3):
                # Check if changing the current cell can make a 2x2 square of the same color
                if (i < 2 and j < 2 and grid[i][j] != grid[i+1][j+1] and grid[i][j+1] == grid[i+1][j]) or \
                   (i < 2 and j > 0 and grid[i][j] != grid[i+1][j-1] and grid[i][j-1] == grid[i+1][j]) or \
                   (i > 0 and j < 2 and grid[i][j] != grid[i-1][j+1] and grid[i-1][j] == grid[i][j+1]) or \
                   (i > 0 and j > 0 and grid[i][j] != grid[i-1][j-1] and grid[i-1][j] == grid[i][j-1]):
                    return True
        return False