class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # First, check if there is already a 2x2 sub-square of the same color.
        # There are 4 possible 2x2 sub-squares in a 3x3 matrix:
        # Top-left corners for these 2x2 squares: (0,0), (0,1), (1,0), (1,1)

        def has_2x2_square_same_color(g):
            # Check all 4 possible 2x2 squares
            for r in range(2):
                for c in range(2):
                    cell_color = g[r][c]
                    if (g[r][c+1] == cell_color and 
                        g[r+1][c] == cell_color and 
                        g[r+1][c+1] == cell_color):
                        return True
            return False

        # If the grid already has a 2x2 sub-square of the same color, return True
        if has_2x2_square_same_color(grid):
            return True

        # Otherwise, try changing at most one cell and see if it creates a 2x2 square
        for r in range(3):
            for c in range(3):
                # Original color of the cell
                original = grid[r][c]
                # Flip color: 'B' -> 'W' or 'W' -> 'B'
                flipped = 'B' if original == 'W' else 'W'
                
                # Change the cell
                grid[r][c] = flipped
                if has_2x2_square_same_color(grid):
                    # Restore and return True
                    grid[r][c] = original
                    return True
                # Restore the cell
                grid[r][c] = original

        # If no single change can create a 2x2 sub-square of the same color, return False
        return False