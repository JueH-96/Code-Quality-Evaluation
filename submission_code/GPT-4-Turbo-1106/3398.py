class Solution:
    def canMakeSquare(self, grid: list[list[str]]) -> bool:
        # Helper function to check if a 2x2 square is of the same color
        def is_same_color_square(x, y):
            return grid[x][y] == grid[x+1][y] == grid[x][y+1] == grid[x+1][y+1]
        
        # Check all 2x2 squares in the grid
        for i in range(2):
            for j in range(2):
                if is_same_color_square(i, j):
                    return True
        
        # Check if changing one cell can create a 2x2 square of the same color
        for i in range(2):
            for j in range(2):
                # Save the original colors
                original = grid[i][j], grid[i+1][j], grid[i][j+1], grid[i+1][j+1]
                # Try changing each of the four cells and check for a square
                for k in range(4):
                    grid[i + k//2][j + k%2] = 'W' if original[k] == 'B' else 'B'
                    if is_same_color_square(i, j):
                        return True
                    # Revert the change
                    grid[i + k//2][j + k%2] = original[k]
        
        return False