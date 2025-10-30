class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def check_square(i, j):
            colors = set(grid[x][y] for x in range(i, i+2) for y in range(j, j+2))
            return len(colors) == 1 or len(colors) == 2

        # Check if there's already a 2x2 square of the same color
        for i in range(2):
            for j in range(2):
                if len(set(grid[x][y] for x in range(i, i+2) for y in range(j, j+2))) == 1:
                    return True

        # Check if changing one cell can create a 2x2 square
        for i in range(3):
            for j in range(3):
                original = grid[i][j]
                grid[i][j] = 'W' if original == 'B' else 'B'
                
                if (i < 2 and j < 2 and check_square(i, j)) or \
                   (i < 2 and j > 0 and check_square(i, j-1)) or \
                   (i > 0 and j < 2 and check_square(i-1, j)) or \
                   (i > 0 and j > 0 and check_square(i-1, j-1)):
                    return True
                
                grid[i][j] = original

        return False